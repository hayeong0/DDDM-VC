import math
import torch 
import numpy as np
from torch.nn import functional as F

from model.base import BaseModule
from model.diffusion_module import *


class GradLogPEstimator(BaseModule):
    def __init__(self, dim_base, dim_cond, dim_mults=(1, 2, 4)):
        super(GradLogPEstimator, self).__init__()

        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4),
                                       Mish(), torch.nn.Linear(dim_base * 4, dim_base))
        cond_total = dim_base + 256
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                              Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])

        num_resolutions = len(in_out) 

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]  

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in)]))
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(self, x, x_mask, enc_out, spk, t):
        condition = self.time_pos_emb(t) 
        t = self.mlp(condition) 

        x = torch.stack([enc_out, x], 1)
        x_mask = x_mask.unsqueeze(1)

        condition = torch.cat([condition, spk.squeeze(2)], 1) 
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)  

        condition = torch.cat(x.shape[2] * [condition], 2)  
        condition = torch.cat(x.shape[3] * [condition], 3)
        x = torch.cat([x, condition], 1)

        hiddens = []
        masks = [x_mask]

        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)
        return (output * x_mask).squeeze(1)

class Diffusion(BaseModule):
    def __init__(self, n_feats, dim_unet, dim_spk, beta_min, beta_max):
        super(Diffusion, self).__init__()
        self.estimator_src = GradLogPEstimator(dim_unet, dim_spk)
        self.estimator_ftr = GradLogPEstimator(dim_unet, dim_spk)

        self.n_feats = n_feats
        self.dim_unet = dim_unet
        self.dim_spk = dim_spk
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (t + s)
        beta_integral *= (t - s)
        if use_torch:
            gamma = torch.exp(-0.5 * p * beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5 * p * beta_integral)
        return gamma

    def get_mu(self, s, t):
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(self, s, t):
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(self, s, t):
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def compute_diffused_mean(self, x0, mask, src_out, ftr_out, t, use_torch=False):

        x0_weight = self.get_gamma(0, t, use_torch=use_torch)  
        mean_weight = 1.0 - x0_weight
        xt_src = x0 * x0_weight + src_out * mean_weight
        xt_ftr = x0 * x0_weight + ftr_out * mean_weight
        return xt_src * mask, xt_ftr * mask

    def forward_diffusion(self, x0, mask, src_out, ftr_out, t):
        xt_src, xt_ftr = self.compute_diffused_mean(x0, mask, src_out, ftr_out, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt_src = xt_src + z * torch.sqrt(variance)
        xt_ftr = xt_ftr + z * torch.sqrt(variance)

        return xt_src * mask, xt_ftr * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z_src, z_ftr, mask, src_out, ftr_out, spk,
                          n_timesteps, mode):
        h = 1.0 / n_timesteps
        xt_src = z_src * mask
        xt_ftr = z_ftr * mask
        for i in range(n_timesteps):
            t = 1.0 - i * h
            time = t * torch.ones(z_src.shape[0], dtype=z_src.dtype, device=z_src.device)
            beta_t = self.get_beta(t)

            if mode == 'ml':
                kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                kappa /= (self.get_gamma(0, t) * beta_t * h)
                kappa -= 1.0
                omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                omega += self.get_mu(t - h, t)
                omega -= (0.5 * beta_t * h + 1.0)
                sigma = self.get_sigma(t - h, t)

            else:
                kappa = 0.0
                omega = 0.0
                sigma = math.sqrt(beta_t * h)

            dxt_src = (src_out - xt_src) * (0.5 * beta_t * h + omega)
            dxt_ftr = (ftr_out - xt_ftr) * (0.5 * beta_t * h + omega)

            estimated_score = (self.estimator_src(xt_src, mask, src_out, spk, time) +
                               self.estimator_ftr(xt_ftr, mask, ftr_out, spk, time)) \
                              * (1.0 + kappa) * (beta_t * h)     
            dxt_src -= estimated_score
            dxt_ftr -= estimated_score
            
            sigma_n = torch.randn_like(z_src, device=z_src.device) * sigma
            dxt_src += sigma_n
            dxt_ftr += sigma_n

            xt_src = (xt_src - dxt_src) * mask
            xt_ftr = (xt_ftr - dxt_ftr) * mask

        return xt_src, xt_ftr

    @torch.no_grad()
    def forward(self, z_src, z_ftr, mask, src_out, ftr_out, spk, n_timesteps, mode):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z_src, z_ftr

        return self.reverse_diffusion(z_src, z_ftr, mask, src_out, ftr_out, spk, n_timesteps, mode)

    def loss_t(self, x0, mask, src_out, ftr_out, spk, t):
        xt_src, xt_ftr, z = self.forward_diffusion(x0, mask, src_out, ftr_out, t)

        z_estimation = self.estimator_src(xt_src, mask, src_out, spk, t)
        z_estimation += self.estimator_ftr(xt_ftr, mask, ftr_out, spk, t)

        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)

        return loss

    def compute_loss(self, x0, mask, src_out, ftr_out, spk, offset=1e-5):
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.loss_t(x0, mask, src_out, ftr_out, spk, t)
