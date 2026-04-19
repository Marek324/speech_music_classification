# tests/test_nn_tcn_augmentation.py
# Tests for mel-space augmentation (waveform-domain ops were dropped during the
# mel-precompute refactor — paper §3.4 applies augmentation on saved spectrograms).

import math

import pytest
import torch

from src.nn.tcn.augmentation import _LN10_OVER_10, augment_mel, random_gain_mel


@pytest.fixture
def mel_batch():
    torch.manual_seed(0)
    return torch.randn(4, 80, 30)


@pytest.fixture
def norm_std():
    # Deterministic per-bin std > 0.
    return torch.ones(1, 80, 1) * 2.0


# ---------------------------------------------------------------------------
# random_gain_mel
# ---------------------------------------------------------------------------


def test_random_gain_mel_preserves_shape(mel_batch, norm_std):
    out = random_gain_mel(mel_batch, norm_std)
    assert out.shape == mel_batch.shape
    assert out.dtype == mel_batch.dtype


def test_random_gain_mel_applies_additive_shift(mel_batch, norm_std):
    """Gain is applied as an additive shift in normalized log-mel space."""
    torch.manual_seed(1)
    out = random_gain_mel(mel_batch, norm_std, min_db=3.0, max_db=3.0)  # deterministic: 3 dB
    expected_shift = (3.0 * _LN10_OVER_10) / 2.0  # norm_std == 2 everywhere
    diff = out - mel_batch
    # All entries should shift by the same value (per-example broadcast).
    assert torch.allclose(diff, torch.full_like(diff, expected_shift), atol=1e-6)


def test_random_gain_mel_is_per_example():
    """Each batch element should receive an independent gain."""
    torch.manual_seed(2)
    mel = torch.zeros(32, 80, 10)
    norm_std = torch.ones(1, 80, 1)
    out = random_gain_mel(mel, norm_std, min_db=-6.0, max_db=6.0)
    # If gain were shared across batch, all per-example shifts would be equal.
    per_example_shift = out[:, 0, 0]
    assert per_example_shift.unique().numel() > 1


def test_random_gain_mel_db_bounds(norm_std):
    """Shift magnitude must stay within the dB range (converted to log-mel shift)."""
    torch.manual_seed(3)
    mel = torch.zeros(200, 80, 1)
    out = random_gain_mel(mel, norm_std, min_db=-6.0, max_db=6.0)
    max_expected_shift = (6.0 * _LN10_OVER_10) / 2.0
    assert out.abs().max().item() <= max_expected_shift + 1e-6


# ---------------------------------------------------------------------------
# augment_mel
# ---------------------------------------------------------------------------


class _FakeFrontend:
    """Minimal stand-in providing ``norm_std`` — avoids full frontend setup in tests."""

    def __init__(self, n_mels: int):
        self.norm_std = torch.ones(1, n_mels, 1)


def test_augment_mel_shape_preserved(mel_batch):
    fe = _FakeFrontend(n_mels=mel_batch.shape[1])
    out = augment_mel(mel_batch, fe, p=1.0)
    assert out.shape == mel_batch.shape


def test_augment_mel_p_zero_is_noop(mel_batch):
    """p=0 must never modify the input."""
    fe = _FakeFrontend(n_mels=mel_batch.shape[1])
    out = augment_mel(mel_batch, fe, p=0.0)
    assert torch.equal(out, mel_batch)


def test_augment_mel_p_one_always_shifts(mel_batch):
    """p=1.0 must always return a shifted copy (not a no-op)."""
    fe = _FakeFrontend(n_mels=mel_batch.shape[1])
    torch.manual_seed(4)
    out = augment_mel(mel_batch, fe, p=1.0)
    # Possible but vanishingly unlikely that gain samples exactly 0 dB in float32.
    assert not torch.equal(out, mel_batch)
