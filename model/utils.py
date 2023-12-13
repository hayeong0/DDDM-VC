

import os
import glob
import sys
import logging
import torch
import torchaudio
import numpy as np

from librosa.filters import mel as librosa_mel_fn
from model.base import BaseModule

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def mse_loss(x, y, mask, n_feats):
    loss = torch.sum(((x - y)**2) * mask)
    return loss / (torch.sum(mask) * n_feats)

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1

class PseudoInversion(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft):
        super(PseudoInversion, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        mel_basis = librosa_mel_fn(sampling_rate, n_fft, n_mels, 0, 8000)
        mel_basis_inverse = np.linalg.pinv(mel_basis)
        mel_basis_inverse = torch.from_numpy(mel_basis_inverse).float()
        self.register_buffer("mel_basis_inverse", mel_basis_inverse)

    def forward(self, log_mel_spectrogram):
        mel_spectrogram = torch.exp(log_mel_spectrogram)
        stftm = torch.matmul(self.mel_basis_inverse, mel_spectrogram)
        return stftm

class InitialReconstruction(BaseModule):
    def __init__(self, n_fft, hop_size):
        super(InitialReconstruction, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)

    def forward(self, stftm):
        real_part = torch.ones_like(stftm, device=stftm.device)
        imag_part = torch.zeros_like(stftm, device=stftm.device)
        stft = torch.stack([real_part, imag_part], -1)*stftm.unsqueeze(-1)
        istft = torchaudio.functional.istft(stft, n_fft=self.n_fft, 
                           hop_length=self.hop_size, win_length=self.n_fft, 
                           window=self.window, center=True)
        return istft.unsqueeze(1)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration