# tcn/preprocess.py
# Log-mel spectrogram frontend for TCN.

import torch
import torch.nn as nn
import torchaudio.transforms as T

from .config import get_config


class LogMelSpectrogram(nn.Module):
    """Converts raw waveform to log-mel spectrogram."""

    def __init__(self, sample_rate=None, n_fft=None, hop_length=None, n_mels=None, f_min=None, f_max=None):
        super().__init__()
        cfg = get_config()
        sr = sample_rate if sample_rate is not None else cfg["sample_rate"]
        self.mel = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft or cfg["n_fft"],
            hop_length=hop_length or cfg["hop_length"],
            n_mels=n_mels or cfg["n_mels"],
            f_min=f_min if f_min is not None else cfg["f_min"],
            f_max=f_max if f_max is not None else cfg["f_max"],
        )

    def forward(self, waveform):
        mel = self.mel(waveform)
        log_mel = torch.log(mel + 1e-7)
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-7
        return (log_mel - mean) / std
