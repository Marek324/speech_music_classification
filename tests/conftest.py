# tests/conftest.py
# Shared fixtures for all test modules.

import numpy as np
import pytest
import torch
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.toml"
ABLATION_CFG_PATH = ROOT / "src" / "exp" / "tcn_ablation" / "config.toml"


# ── global config singleton reset ──────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_global_config():
    """Reset src.config._cfg singleton before and after every test."""
    import src.config as _cfg_mod
    _cfg_mod._cfg = None
    yield
    _cfg_mod._cfg = None


@pytest.fixture
def classic_config_dt():
    """Initialize global config for decision_tree model."""
    import src.config as _cfg_mod
    _cfg_mod._cfg = None
    _cfg_mod.init_config(CFG_PATH, "decision_tree")
    yield _cfg_mod.get_config()
    _cfg_mod._cfg = None


@pytest.fixture
def classic_config_gmm():
    """Initialize global config for gmm model."""
    import src.config as _cfg_mod
    _cfg_mod._cfg = None
    _cfg_mod.init_config(CFG_PATH, "gmm")
    yield _cfg_mod.get_config()
    _cfg_mod._cfg = None


@pytest.fixture
def classic_config_svm():
    """Initialize global config for svm model."""
    import src.config as _cfg_mod
    _cfg_mod._cfg = None
    _cfg_mod.init_config(CFG_PATH, "svm")
    yield _cfg_mod.get_config()
    _cfg_mod._cfg = None


# ── audio fixtures ─────────────────────────────────────────────────────────
SR = 22050
HOP = 512
N_FFT = 1024
N_MELS = 80


@pytest.fixture
def sr():
    return SR


@pytest.fixture
def hop():
    return HOP


@pytest.fixture
def n_fft():
    return N_FFT


@pytest.fixture
def mono_wav():
    """3-second mono waveform (1, N) tensor at 22050 Hz."""
    return torch.randn(1, SR * 3)


# ── tiny TCN config ─────────────────────────────────────────────────────────
TINY_TCN_CFG = {
    "sample_rate": SR,
    "n_fft": N_FFT,
    "hop_length": HOP,
    "n_mels": N_MELS,
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


@pytest.fixture
def tiny_tcn_cfg():
    return dict(TINY_TCN_CFG)


def make_detector(cfg=None):
    """Build a SpeechMusicDetector with injected normalization stats."""
    from src.nn.tcn.model import SpeechMusicDetector
    if cfg is None:
        cfg = TINY_TCN_CFG
    model = SpeechMusicDetector(cfg=cfg, stats_path=Path("/nonexistent"))
    model.fe.norm_mean = torch.zeros(1, cfg["n_mels"], 1)
    model.fe.norm_std = torch.ones(1, cfg["n_mels"], 1)
    model.fe._stats_loaded = True
    return model


@pytest.fixture
def detector(tiny_tcn_cfg):
    return make_detector(tiny_tcn_cfg)
