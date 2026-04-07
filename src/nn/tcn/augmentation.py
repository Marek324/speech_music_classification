# tcn/augmentation.py
# Waveform augmentation applied during training.
# Based on Schlüter & Grill (2015) as cited in Lemaire & Holzapfel (2019) §3.4.

import torch


def random_gain(wav: torch.Tensor, min_db: float = -6.0, max_db: float = 6.0) -> torch.Tensor:
    """Multiply waveform by a random gain uniformly sampled in [min_db, max_db] dB."""
    gain_db = torch.empty(wav.shape[0], 1, device=wav.device).uniform_(min_db, max_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return wav * gain_linear


def add_gaussian_noise(wav: torch.Tensor, snr_db: float = 30.0) -> torch.Tensor:
    """Add Gaussian noise at a fixed SNR (dB) relative to each waveform's RMS."""
    rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-9)
    noise_rms = rms * (10.0 ** (-snr_db / 20.0))
    noise = torch.randn_like(wav) * noise_rms
    return wav + noise


def augment(wav: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Apply random gain and Gaussian noise to a batch of waveforms.

    Args:
        wav: (B, samples)
        p:   probability of applying each augmentation independently
    Returns:
        augmented wav, same shape
    """
    if torch.rand(1).item() < p:
        wav = random_gain(wav)
    if torch.rand(1).item() < p:
        wav = add_gaussian_noise(wav)
    return wav
