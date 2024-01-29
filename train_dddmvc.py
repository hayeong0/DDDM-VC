import os
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import random
import commons
import utils

from augmentation.aug import Augment
from model_f0_vqvae import Quantizer
from model.vc_dddm_mixup import Wav2vec2, DDDM
from data_loader import AudioDataset, MelSpectrogramFixed
from hifigan.vocoder import HiFi
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
global_step = 0

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda(rank)

    train_dataset = AudioDataset(hps, training=True)
    train_sampler = DistributedSampler(train_dataset) if n_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=hps.train.batch_size, num_workers=32,
        sampler=train_sampler, drop_last=True, persistent_workers=True, pin_memory=True
    )

    if rank == 0:
        test_dataset = AudioDataset(hps, training=False)
        eval_loader = DataLoader(test_dataset, batch_size=1)

        net_v = HiFi(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cuda()
        path_ckpt = './hifigan/G_2930000.pth'

        utils.load_checkpoint(path_ckpt, net_v, None)
        net_v.eval()
        net_v.dec.remove_weight_norm()
    else:
        net_v = None
        
    w2v = Wav2vec2().cuda(rank)
    aug = Augment(hps).cuda(rank)

    model = DDDM(hps.data.n_mel_channels, hps.diffusion.spk_dim,
                   hps.diffusion.dec_dim, hps.diffusion.beta_min, hps.diffusion.beta_max, hps).cuda()

    f0_quantizer = Quantizer(hps).cuda(rank)
    utils.load_checkpoint('./f0_vqvae/f0_vqvae.pth', f0_quantizer)
    f0_quantizer.eval() 
     
    if rank == 0:
        num_param = get_param_num(model.encoder)
        print('[Encoder] number of Parameters:', num_param)
        num_param = get_param_num(model.decoder)
        print('[Decoder] number of Parameters:', num_param)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    model = DDP(model, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), model, optimizer) 
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [model, mel_fn, w2v, f0_quantizer, aug, net_v], optimizer,
                               scheduler_g, scaler, [train_loader, eval_loader], logger, [writer, writer_eval], n_gpus)
        else:
            train_and_evaluate(rank, epoch, hps, [model, mel_fn, w2v, f0_quantizer, aug, net_v], optimizer,
                               scheduler_g, scaler, [train_loader, None], None, None, n_gpus)
        scheduler_g.step()

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, n_gpus):
    model, mel_fn, w2v, f0_quantizer, aug, net_v = nets
    optimizer = optims
    scheduler_g = schedulers
    train_loader, eval_loader = loaders

    if writers is not None:
        writer, writer_eval = writers

    global global_step
    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)
    
    model.train()
    for batch_idx, (x, x_f0, length) in enumerate(train_loader):
        x = x.cuda(rank, non_blocking=True)
        x_f0 = x_f0.cuda(rank, non_blocking=True)
        length = length.cuda(rank, non_blocking=True).squeeze()

        mel_x = mel_fn(x)
        aug_x = aug(x)
        nan_x = torch.isnan(aug_x).any()
        x = x if nan_x else aug_x
        x_pad = F.pad(x, (40, 40), "reflect")
        
        w2v_x = w2v(x_pad)
        f0_x = f0_quantizer.code_extraction(x_f0)

        optimizer.zero_grad()
        loss_diff, loss_mel = model.module.compute_loss(mel_x, w2v_x, f0_x, length, hps.model.mixup_ratio)

        loss_gen_all = loss_diff + loss_mel*hps.train.c_mel

        if hps.train.fp16_run:
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optimizer)
            grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_gen_all.backward()
            grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
            optimizer.step()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                losses = [loss_diff]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/diff": loss_diff, "loss/g/mel": loss_mel})

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                torch.cuda.empty_cache()
                evaluate(hps, model, mel_fn, w2v, f0_quantizer, net_v, eval_loader, writer_eval)

                if global_step % hps.train.save_interval == 0:
                    utils.save_checkpoint(model, optimizer, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, model, mel_fn, w2v, f0_quantizer, net_v, eval_loader, writer_eval):
    model.eval()
    image_dict = {}
    audio_dict = {}
    mel_loss = 0
    enc_loss = 0
    with torch.no_grad():
        for batch_idx, (y, y_f0) in enumerate(eval_loader):
            y = y.cuda(0)
            y_f0 = y_f0.cuda(0)

            mel_y = mel_fn(y)
            f0_y = f0_quantizer.code_extraction(y_f0)
            length = torch.LongTensor([mel_y.size(2)]).cuda(0)

            y_pad = F.pad(y, (40, 40), "reflect")
            w2v_y = w2v(y_pad)

            enc_output, src_out, ftr_out, mel_rec = model(mel_y, w2v_y, f0_y, length, n_timesteps=6, mode='ml')

            mel_loss += F.l1_loss(mel_y, mel_rec).item()
            enc_loss += F.l1_loss(mel_y, enc_output).item()

            if batch_idx > 100:
                break
            if batch_idx <= 4:
                y_hat = net_v(mel_rec)
                enc_hat = net_v(enc_output)
                src_hat = net_v(src_out)
                ftr_hat = net_v(ftr_out)

                plot_mel = torch.cat([mel_y, mel_rec, enc_output, ftr_out, src_out], dim=1)
                plot_mel = plot_mel.clip(min=-10, max=10)

                image_dict.update({
                    "gen/mel_{}".format(batch_idx): utils.plot_spectrogram_to_numpy(plot_mel.squeeze().cpu().numpy())
                })
                audio_dict.update({
                    "gen/audio_{}".format(batch_idx): y_hat.squeeze(),
                    "gen/enc_audio_{}".format(batch_idx): enc_hat.squeeze(),
                    "gen/src_audio_{}".format(batch_idx): src_hat.squeeze(),
                    "gen/ftr_audio_{}".format(batch_idx): ftr_hat.squeeze(),
                })
                if global_step == 0:
                    audio_dict.update({"gt/audio_{}".format(batch_idx): y.squeeze()})

        mel_loss /= 100
        enc_loss /= 100

    scalar_dict = {"val/mel": mel_loss, "val/enc_mel": enc_loss}
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict
    )
    model.train()


if __name__ == "__main__":
    main()
