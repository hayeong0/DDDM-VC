import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from .peq import ParametricEqualizer

class Augment(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.config = h
        self.coder = LinearPredictiveCoding(
            32, h.data.win_length, h.data.hop_length)
        self.peq = ParametricEqualizer(
            h.data.sampling_rate, h.data.win_length)
        self.register_buffer(
            'window',
            torch.hann_window(h.data.win_length),
            persistent=False)
        f_min, f_max, peaks = 60, 10000, 8
        self.register_buffer(
            'peak_centers',
            f_min * (f_max / f_min) ** (torch.arange(peaks) / (peaks - 1)),
            persistent=False)
 
    def forward(self, 
                wavs: torch.Tensor,
                mode: str = 'linear',
               ):
        """Augment the audio signal, random pitch, formant shift and PEQ.
        Args:
            wavs: [torch.float32; [B, T]], audio signal.
            mode: interpolation mode, `linear` or `nearest`.
        """
        auxs = {}
        fft = torch.stft(
            wavs,
            self.config.data.filter_length,
            self.config.data.hop_length,
            self.config.data.win_length,
            self.window,
            return_complex=True)
        
        power, gain = self.sample(wavs) # for fs, ps
        
        if power is not None:
            q_min, q_max = 2, 5
            q = q_min * (q_max / q_min) ** power
            
            if gain is None:
                gain = torch.zeros_like(q[:, :-2])
                
            bsize = wavs.shape[0] 
            center = self.peak_centers[None].repeat(bsize, 1) 
            peaks = torch.prod(
                self.peq.peaking_equalizer(center, gain, q[:, :-2]), dim=1)
            lowpass = self.peq.low_shelving(60, q[:, -2])
            highpass = self.peq.high_shelving(10000, q[:, -1])
            
            filters = peaks * highpass * lowpass
            fft = fft * filters[..., None]
            auxs.update({'peaks': peaks, 'highpass': highpass, 'lowpass': lowpass})
    
        # Formant shifting and Pitch shifting
        fs_ratio = 1.4 
        ps_ratio = 2.0
       
        code = self.coder.from_stft(fft / fft.abs().mean(dim=1)[:, None].clamp_min(1e-7))
        filter_ = self.coder.envelope(code)
        source = fft.transpose(1, 2) / (filter_ + 1e-7)
        
        bsize = wavs.shape[0]
        def sampler(ratio):
            shifts = torch.rand(bsize, device=wavs.device) * (ratio - 1.) + 1.
            flip = torch.rand(bsize) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts
        
        fs_shift = sampler(fs_ratio)
        ps_shift = sampler(ps_ratio)

        source = fft.transpose(1, 2) / (filter_ + 1e-7)
         
        filter_ = self.interp(filter_, fs_shift, mode=mode)
        source = self.interp(source, ps_shift, mode=mode)
       
        fft = (source * filter_).transpose(1, 2)
        out = torch.istft(
            fft,
            self.config.data.filter_length,
            self.config.data.hop_length,
            self.config.data.win_length,
            self.window)
        out = out / out.max(dim=-1, keepdim=True).values.clamp_min(1e-7)
        
        return out
    
    def sample(self, wavs: torch.Tensor):
        bsize, _ = wavs.shape
        
        # parametric equalizer
        peaks = 8
        # quality factor
        power = torch.rand(bsize, peaks + 2, device=wavs.device)
        # gains
        g_min, g_max = -12, 12
        gain = torch.rand(bsize, peaks, device=wavs.device) * (g_max - g_min) + g_min
        
        return power, gain
    
    @staticmethod
    def complex_interp(inputs: torch.Tensor, *args, **kwargs):
        mag = F.interpolate(inputs.abs(), *args, **kwargs)
        angle = F.interpolate(inputs.angle(), *args, **kwargs)
        return torch.polar(mag, angle)

    def interp(self, inputs: torch.Tensor, shifts: torch.Tensor, mode: str):
        """Interpolate the channel axis with dynamic shifts.
        Args:
            inputs: [torch.complex64; [B, T, C]], input tensor.
            shifts: [torch.float32; [B]], shift factor.
            mode: interpolation mode.
        Returns:
            [torch.complex64; [B, T, C]], interpolated.
        """
        INTERPOLATION = {
            torch.float32: F.interpolate,
            torch.complex64: Augment.complex_interp}
        assert inputs.dtype in INTERPOLATION, 'unsupported interpolation'
        interp_fn = INTERPOLATION[inputs.dtype]
       
        _, _, channels = inputs.shape
       
        interp = [
            interp_fn(
                f[None], scale_factor=s.item(), mode=mode)[..., :channels]
            for f, s in zip(inputs, shifts)]
       
        return torch.cat([
            F.pad(f, [0, channels - f.shape[-1]])
            for f in interp], dim=0)


class LinearPredictiveCoding(nn.Module):
    """LPC: Linear-predictive coding supports.
    """

    def __init__(self, num_code: int, windows: int, strides: int):
        """Initializer.
        Args:
            num_code: the number of the coefficients.
            windows: size of the windows.
            strides: the number of the frames between adjacent windows.
        """
        super().__init__()
        self.num_code = num_code
        self.windows = windows
        self.strides = strides

    def forward(self, inputs: torch.Tensor):
        """Compute the linear-predictive coefficients from inputs.
        Args:
            inputs: [torch.float32; [B, T]], audio signal.
        Returns:
            [torch.float32; [B, T / strides, num_code]], coefficients.
        """
        w = self.windows
        frames = F.pad(inputs, [0, w]).unfold(-1, w, self.strides)
        corrcoef = LinearPredictiveCoding.autocorr(frames)
      
        return LinearPredictiveCoding.solve_toeplitz(
            corrcoef[..., :self.num_code + 1])

    def from_stft(self, inputs: torch.Tensor):
        """Compute the linear-predictive coefficients from STFT.
        Args:
            inputs: [torch.complex64; [B, windows // 2 + 1, T / strides]], fourier features.
        Returns:
            [torch.float32; [B, T / strides, num_code]], linear-predictive coefficient.
        """
        corrcoef = torch.fft.irfft(inputs.abs().square(), dim=1)
       
        return LinearPredictiveCoding.solve_toeplitz(
            corrcoef[:, :self.num_code + 1].transpose(1, 2))

    def envelope(self, lpc: torch.Tensor):
        """LPC to spectral envelope.
        Args:
            lpc: [torch.float32; [..., num_code]], coefficients.
        Returns:
            [torch.float32; [..., windows // 2 + 1]], filters.
        """
        denom = torch.fft.rfft(-F.pad(lpc, [1, 0], value=1.), self.windows, dim=-1).abs()
        # for preventing zero-division
        denom[(denom.abs() - 1e-7) < 0] = 1.
        return denom ** -1

    @staticmethod
    def autocorr(wavs: torch.Tensor):
        """Compute the autocorrelation.
        Args: audio signal.
        Returns: auto-correlation.
        """
        fft = torch.fft.rfft(wavs, dim=-1)
        return torch.fft.irfft(fft.abs().square(), dim=-1)

    @staticmethod
    def solve_toeplitz(corrcoef: torch.Tensor):
        """Solve the toeplitz matrix.
        Args:
            corrcoef: [torch.float32; [..., num_code + 1]], auto-correlation.
        Returns:
            [torch.float32; [..., num_code]], solutions.
        """
      
        solutions = F.pad(
            (-corrcoef[..., 1] / corrcoef[..., 0].clamp_min(1e-7))[..., None],
            [1, 0], value=1.)
        
        extra = corrcoef[..., 0] + corrcoef[..., 1] * solutions[..., 1]

        ## solve residuals
        num_code = corrcoef.shape[-1] - 1
        for k in range(1, num_code):
            lambda_value = (
                                   -solutions[..., :k + 1]
                                   * torch.flip(corrcoef[..., 1:k + 2], dims=[-1])
                           ).sum(dim=-1) / extra.clamp_min(1e-7)
            aug = F.pad(solutions, [0, 1])
            solutions = aug + lambda_value[..., None] * torch.flip(aug, dims=[-1])
            extra = (1. - lambda_value ** 2) * extra
       
        return solutions[..., 1:]