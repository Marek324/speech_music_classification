# tests/test_nn_tcn_streaming.py

import torch
import pytest
from pathlib import Path
from src.nn.tcn.streaming import StreamingInference


SR = 22050
HOP = 512

TINY_CFG = {
    "sample_rate": SR,
    "n_fft": 1024,
    "hop_length": HOP,
    "n_mels": 80,
    "f_min": 27.5,
    "f_max": 8000.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "seq_len": 128,
    "dataset": {"url": None, "name": "full"},
    "model": {
        "n_filters": 8,
        "kernel_size": 3,
        "n_layers": 2,
        "n_stacks": 1,
        "dropout": 0.0,
        "n_classes": 3,
        "use_weight_norm": False,
    },
}


def _make_streaming():
    from src.nn.tcn.model import SpeechMusicDetector
    model = SpeechMusicDetector(cfg=TINY_CFG, stats_path=Path("/nonexistent"))
    model.fe.norm_mean = torch.zeros(1, TINY_CFG["n_mels"], 1)
    model.fe.norm_std = torch.ones(1, TINY_CFG["n_mels"], 1)
    model.fe._stats_loaded = True
    model.eval()
    return StreamingInference(model, hop_length=HOP, sample_rate=SR)


# ── short chunk (below threshold) ────────────────────────────────────────

def test_process_chunk_short_returns_zeros():
    si = _make_streaming()
    short_chunk = torch.randn(HOP * 2)  # < HOP * 4 min threshold
    result = si.process_chunk(short_chunk)
    assert result == (0.0, 0.0, 0.0)


# ── long enough chunk ─────────────────────────────────────────────────────

def test_process_chunk_output_is_three_floats():
    si = _make_streaming()
    chunk = torch.randn(HOP * 8)
    result = si.process_chunk(chunk)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_process_chunk_output_range():
    si = _make_streaming()
    chunk = torch.randn(HOP * 8)
    speech, music, inactive = si.process_chunk(chunk)
    for v in (speech, music, inactive):
        assert 0.0 <= v <= 1.0


# ── buffer management ─────────────────────────────────────────────────────

def test_process_chunk_buffer_stays_bounded():
    si = _make_streaming()
    max_allowed = si.left_rf_samples + HOP * 8 + 10  # small slack
    chunk = torch.randn(HOP * 8)
    for _ in range(20):
        si.process_chunk(chunk)
        assert si.buffer.shape[-1] <= max_allowed


def test_process_chunk_buffer_trimmed_after_processing():
    si = _make_streaming()
    chunk = torch.randn(HOP * 8)
    si.process_chunk(chunk)
    # After processing, buffer should be <= left_rf_samples
    assert si.buffer.shape[-1] <= si.left_rf_samples
