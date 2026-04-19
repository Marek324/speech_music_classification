# tests/test_nn_tcn_model.py

import torch
import pytest
from pathlib import Path
from src.nn.tcn.model import CausalTCN, SpeechMusicDetector


TINY_CFG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 512,
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


def _make_detector(cfg=None):
    cfg = cfg or TINY_CFG
    model = SpeechMusicDetector(cfg=cfg, stats_path=Path("/nonexistent"))
    model.fe.norm_mean = torch.zeros(1, cfg["n_mels"], 1)
    model.fe.norm_std = torch.ones(1, cfg["n_mels"], 1)
    model.fe._stats_loaded = True
    return model


# ── CausalTCN ─────────────────────────────────────────────────────────────

def test_causal_tcn_output_shape():
    tcn = CausalTCN(cfg=TINY_CFG)
    x = torch.randn(1, 80, 50)
    out = tcn(x)
    assert out.shape == (1, 3, 50)


def test_causal_tcn_output_range():
    """CausalTCN emits raw logits. Sigmoid(logits) must land in [0, 1]."""
    tcn = CausalTCN(cfg=TINY_CFG)
    x = torch.randn(2, 80, 100)
    with torch.no_grad():
        out = torch.sigmoid(tcn(x))
    assert (out >= 0).all() and (out <= 1).all()


def test_causal_tcn_small_config():
    cfg = dict(TINY_CFG)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["n_filters"] = 4
    cfg["model"]["n_layers"] = 1
    cfg["model"]["n_stacks"] = 1
    tcn = CausalTCN(cfg=cfg)
    out = tcn(torch.randn(1, 80, 30))
    assert out.shape == (1, 3, 30)


def test_causal_tcn_batch_size():
    tcn = CausalTCN(cfg=TINY_CFG)
    out = tcn(torch.randn(4, 80, 50))
    assert out.shape == (4, 3, 50)


def test_causal_tcn_causality():
    """Output at time t must not depend on input at time t+1 or later."""
    torch.manual_seed(0)
    tcn = CausalTCN(cfg=TINY_CFG)
    tcn.eval()

    x = torch.randn(1, 80, 60)
    with torch.no_grad():
        out_full = tcn(x)

    t = 30
    x_masked = x.clone()
    x_masked[:, :, t:] = 0.0
    with torch.no_grad():
        out_masked = tcn(x_masked)

    assert torch.allclose(out_full[:, :, :t], out_masked[:, :, :t], atol=1e-5)


# ── SpeechMusicDetector ───────────────────────────────────────────────────

def test_speech_music_detector_output_shape():
    model = _make_detector()
    model.eval()
    wav = torch.randn(1, 22050 * 2)
    with torch.no_grad():
        out = model(wav)
    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[1] == 3


def test_speech_music_detector_output_range():
    model = _make_detector()
    model.eval()
    wav = torch.randn(1, 22050 * 2)
    with torch.no_grad():
        out = model(wav)
    assert (out >= 0).all() and (out <= 1).all()


def test_speech_music_detector_no_stats_raises():
    cfg = dict(TINY_CFG)
    model = SpeechMusicDetector(cfg=cfg, stats_path=Path("/nonexistent"))
    # stats not injected → forward should raise RuntimeError
    with pytest.raises(RuntimeError, match="normalization stats"):
        model(torch.randn(1, 22050))
