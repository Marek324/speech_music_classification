# tests/test_nn_tcn_augmentation.py

import math
import torch
import pytest
from src.nn.tcn.augmentation import random_gain, add_gaussian_noise, augment


def test_random_gain_shape():
    wav = torch.randn(4, 22050)
    out = random_gain(wav)
    assert out.shape == wav.shape


def test_random_gain_within_db_range():
    # Run many times; gain must stay within [10^(-6/20), 10^(6/20)]
    min_linear = 10 ** (-6.0 / 20.0)
    max_linear = 10 ** (6.0 / 20.0)
    torch.manual_seed(42)
    wav = torch.ones(1, 1000)
    for _ in range(200):
        out = random_gain(wav)
        ratio = out[0, 0].item()
        assert min_linear - 1e-6 <= ratio <= max_linear + 1e-6


def test_add_gaussian_noise_shape():
    wav = torch.randn(2, 16000)
    out = add_gaussian_noise(wav)
    assert out.shape == wav.shape


def test_add_gaussian_noise_snr():
    # SNR should be approximately 30 dB
    torch.manual_seed(0)
    wav = torch.randn(1, 44100) * 0.5  # non-trivial signal
    noisy = add_gaussian_noise(wav, snr_db=30.0)
    noise = noisy - wav
    signal_rms = wav.pow(2).mean().sqrt().item()
    noise_rms = noise.pow(2).mean().sqrt().item()
    snr_measured = 20 * math.log10(signal_rms / (noise_rms + 1e-12))
    assert abs(snr_measured - 30.0) < 3.0  # within ±3 dB


def test_augment_shape():
    wav = torch.randn(3, 8000)
    out = augment(wav, p=0.5)
    assert out.shape == wav.shape


def test_augment_p0_identity():
    torch.manual_seed(99)
    wav = torch.randn(2, 8000)
    out = augment(wav, p=0.0)
    assert torch.equal(out, wav)


def test_augment_p1_changes_waveform():
    # With p=1.0 both augmentations always apply; output should differ from input
    torch.manual_seed(7)
    wav = torch.randn(2, 8000)
    out = augment(wav, p=1.0)
    assert not torch.equal(out, wav)
