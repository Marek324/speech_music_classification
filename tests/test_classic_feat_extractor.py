# tests/test_classic_feat_extractor.py

import numpy as np
import pytest
from src.classic.feat_extractor import FeatExtractor


# ── helpers ───────────────────────────────────────────────────────────────

def _frame(n=320, dtype=np.float32):
    rng = np.random.default_rng(0)
    return rng.standard_normal(n).astype(dtype)


def _silent_frame(n=320):
    return np.zeros(n, dtype=np.float32)


# ── decision tree feature set ────────────────────────────────────────────

def test_feature_vector_length_dt(classic_config_dt):
    fe = FeatExtractor()
    # run a few frames to fill buffers, then measure stable length
    for _ in range(5):
        out = fe.extract(_frame())
    assert out.ndim == 1
    assert out.shape[0] == 248


def test_feature_vector_consistent_length_dt(classic_config_dt):
    fe = FeatExtractor()
    lengths = set()
    for _ in range(20):
        out = fe.extract(_frame())
        lengths.add(out.shape[0])
    assert len(lengths) == 1


# ── gmm/svm feature set ──────────────────────────────────────────────────

def test_feature_vector_length_gmm(classic_config_gmm):
    fe = FeatExtractor()
    for _ in range(5):
        out = fe.extract(_frame())
    assert out.ndim == 1
    assert out.shape[0] == 9


def test_feature_vector_consistent_length_gmm(classic_config_gmm):
    fe = FeatExtractor()
    lengths = set()
    for _ in range(20):
        out = fe.extract(_frame())
        lengths.add(out.shape[0])
    assert len(lengths) == 1


# ── reset ─────────────────────────────────────────────────────────────────

def test_reset_clears_signal_buffer(classic_config_dt):
    fe = FeatExtractor()
    for _ in range(10):
        fe.extract(_frame())
    assert len(fe.signal_buffer) > 0
    fe.reset()
    assert len(fe.signal_buffer) == 0


def test_reset_clears_feat_buffer(classic_config_dt):
    fe = FeatExtractor()
    for _ in range(10):
        fe.extract(_frame())
    fe.reset()
    assert len(fe.feat_buffer) == 0


def test_reset_clears_last_fft(classic_config_dt):
    fe = FeatExtractor()
    fe.extract(_frame())
    assert fe.last_fft is not None
    fe.reset()
    assert fe.last_fft is None


def test_reset_clears_last_mfcc(classic_config_dt):
    fe = FeatExtractor()
    fe.extract(_frame())
    assert fe.last_mfcc is not None
    fe.reset()
    assert fe.last_mfcc is None


# ── NaN / Inf safety (_safe helper) ──────────────────────────────────────

def test_safe_replaces_nan(classic_config_dt):
    from src.classic.feat_extractor import _safe
    arr = np.array([1.0, np.nan, 3.0])
    out = _safe(arr)
    assert not np.any(np.isnan(out))
    assert out[0] == 1.0 and out[2] == 3.0


def test_safe_replaces_inf(classic_config_dt):
    from src.classic.feat_extractor import _safe
    arr = np.array([np.inf, -np.inf, 2.0])
    out = _safe(arr)
    assert not np.any(np.isinf(out))


def test_normal_frame_output_finite(classic_config_dt):
    fe = FeatExtractor()
    for _ in range(5):
        out = fe.extract(_frame())
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


# ── spectral flux first-call ─────────────────────────────────────────────

def test_spectral_flux_first_call_is_zero(classic_config_dt):
    fe = FeatExtractor()
    assert fe.last_fft is None
    result = fe._spectral_flux(_frame())
    # First call: last_fft is None → returns 0.0 and sets last_fft
    # (After this call last_fft is populated; result should be 0.0)
    assert result == 0.0
    assert fe.last_fft is not None


# ── mfcc diff first-call ─────────────────────────────────────────────────

def test_mfcc_diff_first_call_is_zero(classic_config_dt):
    fe = FeatExtractor()
    assert fe.last_mfcc is None
    mfccs = np.zeros(10)
    result = fe._mfcc_diff_norm(mfccs)
    assert result == 0.0


# ── low short-time energy ratio ──────────────────────────────────────────

def test_lster_empty_energies(classic_config_dt):
    fe = FeatExtractor()
    result = fe._low_short_time_energy_ratio(np.array([]))
    assert result == 0.0
