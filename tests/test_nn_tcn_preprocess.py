# tests/test_nn_tcn_preprocess.py

import torch
import pytest
from pathlib import Path
from src.nn.tcn.preprocess import LogMelSpectrogram

SR = 22050
N_MELS = 80


def _mel_with_stats():
    """LogMelSpectrogram with injected zero-mean unit-std stats."""
    fe = LogMelSpectrogram(sample_rate=SR, stats_path=Path("/nonexistent"))
    fe.norm_mean = torch.zeros(1, N_MELS, 1)
    fe.norm_std = torch.ones(1, N_MELS, 1)
    fe._stats_loaded = True
    return fe


# ── output shape ─────────────────────────────────────────────────────────

def test_log_mel_output_shape_2d():
    fe = _mel_with_stats()
    wav = torch.randn(1, SR * 3)
    with torch.no_grad():
        out = fe(wav)
    assert out.ndim == 3
    assert out.shape[1] == N_MELS


def test_log_mel_output_shape_batch():
    fe = _mel_with_stats()
    wav = torch.randn(4, SR * 2)
    with torch.no_grad():
        out = fe(wav)
    assert out.shape[0] == 4
    assert out.shape[1] == N_MELS


# ── mono conversion ───────────────────────────────────────────────────────

def test_log_mel_mono_from_1d():
    fe = _mel_with_stats()
    wav_1d = torch.randn(SR * 2)  # no batch dim
    with torch.no_grad():
        out = fe(wav_1d)
    assert out.ndim == 3
    assert out.shape[1] == N_MELS


def test_log_mel_mono_from_stereo_3d():
    fe = _mel_with_stats()
    wav_stereo = torch.randn(1, 2, SR * 2)  # (batch, channels, samples)
    with torch.no_grad():
        out = fe(wav_stereo)
    assert out.ndim == 3
    assert out.shape[1] == N_MELS


def test_log_mel_stereo_same_shape_as_mono():
    fe = _mel_with_stats()
    wav_mono = torch.randn(1, SR * 2)
    wav_stereo = torch.cat([wav_mono, wav_mono], dim=0).unsqueeze(0)  # (1, 2, N)
    with torch.no_grad():
        out_mono = fe(wav_mono)
        out_stereo = fe(wav_stereo)
    assert out_mono.shape == out_stereo.shape


# ── resampling ────────────────────────────────────────────────────────────

def test_log_mel_resampling_44100():
    fe_22k = LogMelSpectrogram(sample_rate=SR, stats_path=Path("/nonexistent"))
    fe_44k = LogMelSpectrogram(sample_rate=44100, stats_path=Path("/nonexistent"))
    # inject same stats for both
    for fe in (fe_22k, fe_44k):
        fe.norm_mean = torch.zeros(1, N_MELS, 1)
        fe.norm_std = torch.ones(1, N_MELS, 1)
        fe._stats_loaded = True
    # 2s at 22050 Hz vs 2s at 44100 Hz should produce same T
    wav_22k = torch.randn(1, SR * 2)
    wav_44k = torch.randn(1, 44100 * 2)
    with torch.no_grad():
        out_22k = fe_22k(wav_22k)
        out_44k = fe_44k(wav_44k)
    assert out_22k.shape[-1] == out_44k.shape[-1]


# ── stats validation ──────────────────────────────────────────────────────

def test_log_mel_no_stats_raises():
    fe = LogMelSpectrogram(sample_rate=SR, stats_path=Path("/nonexistent"))
    with pytest.raises(RuntimeError, match="normalization stats"):
        fe(torch.randn(1, SR))


def test_log_mel_with_zero_mean_unit_std():
    fe = _mel_with_stats()
    wav = torch.randn(1, SR * 2)
    with torch.no_grad():
        out = fe(wav)
    # With zero mean and unit std, output equals raw log-mel (finite)
    assert torch.isfinite(out).all()
