import torch.nn as nn
import torch
from modules_vqvae.jukebox import Encoder, Decoder
from modules_vqvae.vq import Bottleneck

LRELU_SLOPE = 0.1

class Quantizer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.encoder = Encoder(**h.f0_encoder_params)
        self.vq = Bottleneck(**h.f0_vq_params)
        self.decoder = Decoder(**h.f0_decoder_params)

    def forward(self, x):
        f0_h = self.encoder(x)
        _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
        f0 = self.decoder(f0_h_q)

        return f0, f0_commit_losses, f0_metrics

    @torch.no_grad()
    def code_extraction(self, x):
        f0_h = self.encoder(x)
        f0_h = [x.detach() for x in f0_h]
        zs, _, _, _ = self.vq(f0_h)
        zs = [x.detach() for x in zs]

        return zs[0].detach()