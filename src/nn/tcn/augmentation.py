# tcn/augmentation.py
# Mel-space augmentation applied during training.
#
# Paper alignment: Lemaire & Holzapfel (2019) §3.4 applies data augmentation
# on pre-computed spectrograms, not raw waveforms. We follow that path.
#
# Gain is the only augmentation currently implemented; the waveform-domain
# Gaussian noise from the previous revision had no paper counterpart
# (Schlüter & Grill 2015 uses Gaussian *filtering* on spectrograms, not
# additive waveform noise) and was dropped during the mel-precompute refactor.

import math

import torch


_LN10_OVER_10 = math.log(10.0) / 10.0  # dB → nat log-power conversion factor


def random_gain_mel(mel: torch.Tensor, norm_std: torch.Tensor,
                    min_db: float = -6.0, max_db: float = 6.0) -> torch.Tensor:
    """Apply a per-example random gain in dB, expressed as an additive shift
    in normalized log-mel space.

    Equivalence to waveform gain:
        waveform amplitude gain g = 10^(dB/20)
        log-power shift           = ln(g^2) = dB * ln(10)/10
        normalized log-mel shift  = log-power shift / norm_std    (per mel bin)

    Args:
        mel:      (B, n_mels, T)  normalized log-mel spectrogram
        norm_std: (1, n_mels, 1)  per-mel-bin std from preprocess stats
    Returns:
        mel + additive shift, same shape
    """
    b = mel.shape[0]
    gain_db = torch.empty(b, 1, 1, device=mel.device, dtype=mel.dtype).uniform_(min_db, max_db)
    shift = (gain_db * _LN10_OVER_10) / norm_std.to(device=mel.device, dtype=mel.dtype)
    return mel + shift


def augment_mel(mel: torch.Tensor, fe, p: float = 0.5) -> torch.Tensor:
    """Apply random gain to a batch of normalized log-mel spectrograms.

    Args:
        mel: (B, n_mels, T)
        fe:  LogMelSpectrogram instance (used to read norm_std)
        p:   probability of applying gain
    Returns:
        augmented mel, same shape
    """
    if torch.rand(1).item() < p:
        mel = random_gain_mel(mel, fe.norm_std)
    return mel
