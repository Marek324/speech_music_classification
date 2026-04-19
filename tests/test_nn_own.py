# tests/test_nn_own.py
# Sanity tests for the own/ module: config, paths, model instantiation + forward.

from pathlib import Path

import pytest
import torch

from src.nn.own.config import (
    _OWN_DEFAULT,
    get_own_config,
    get_own_stats_path,
    get_own_weights_path,
)
from src.nn.own.model import OwnModel


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_own_default_matches_delta2_conv1d():
    """The hardcoded default is the combined-experiment winner."""
    assert _OWN_DEFAULT["frontend"] == "log_mel_delta2"
    assert _OWN_DEFAULT["model"]["preprocessor"] == "conv1d"
    assert _OWN_DEFAULT["model"]["backbone"] == "tcn"
    assert _OWN_DEFAULT["model"]["n_filters"] == 16
    assert _OWN_DEFAULT["model"]["n_layers"] == 4
    assert _OWN_DEFAULT["model"]["n_stacks"] == 3
    assert _OWN_DEFAULT["optimizer"] == "sgd"
    assert _OWN_DEFAULT["name"] == "own"


def test_get_own_config_is_deep_copy():
    """Mutating the returned cfg must not affect subsequent calls."""
    cfg1 = get_own_config()
    cfg1["lr"] = 999.0
    cfg1["model"]["n_filters"] = 999
    cfg2 = get_own_config()
    assert cfg2["lr"] != 999.0
    assert cfg2["model"]["n_filters"] != 999


def test_get_own_config_no_overrides_returns_defaults():
    cfg = get_own_config()
    assert cfg["frontend"] == "log_mel_delta2"
    assert cfg["model"]["preprocessor"] == "conv1d"


def test_get_own_config_applies_top_level_override():
    cfg = get_own_config({"lr": 5e-4, "optimizer": "adam"})
    assert cfg["lr"] == 5e-4
    assert cfg["optimizer"] == "adam"
    # Unchanged defaults still there.
    assert cfg["frontend"] == "log_mel_delta2"


def test_get_own_config_applies_model_override():
    cfg = get_own_config({"n_filters": 32, "dropout": 0.25})
    assert cfg["model"]["n_filters"] == 32
    assert cfg["model"]["dropout"] == 0.25
    # Other model defaults preserved.
    assert cfg["model"]["n_layers"] == 4


def test_get_own_config_mixed_overrides():
    cfg = get_own_config({"lr": 1e-2, "n_filters": 32, "tail": "gru", "tail_width": 32})
    assert cfg["lr"] == 1e-2
    assert cfg["model"]["n_filters"] == 32
    assert cfg["model"]["tail"] == "gru"
    assert cfg["model"]["tail_width"] == 32


def test_get_own_config_ignores_unknown_keys():
    """Keys that aren't in _TOP_KEYS or _MODEL_KEYS should be silently skipped."""
    cfg = get_own_config({"nonsense_flag": 42})
    assert "nonsense_flag" not in cfg
    assert "nonsense_flag" not in cfg["model"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_own_weights_path_has_expected_suffix():
    p = get_own_weights_path()
    assert p.name == "tcn_own.safetensors"
    assert p.parent.name == "own"           # subdir scoping
    assert p.parent.parent.name == "weights"


def test_own_stats_path_has_expected_suffix():
    p = get_own_stats_path()
    assert p.name == "tcn_own_preprocess_stats.pt"
    assert p.parent.name == "own"
    assert p.parent.parent.name == "weights"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _inject_stats(model, cfg):
    model.fe.norm_mean = torch.zeros(1, model.fe.n_features, 1)
    model.fe.norm_std = torch.ones(1, model.fe.n_features, 1)
    model.fe._stats_loaded = True
    return model


def test_own_model_instantiates_with_defaults():
    model = OwnModel(stats_path=Path("/nonexistent"))
    assert model.fe.n_features == 240  # log_mel_delta2 at n_mels=80 -> 3*80
    # No tail by default.
    assert model.tail is None


def test_own_model_forward_shape():
    cfg = get_own_config()
    model = OwnModel(cfg=cfg, stats_path=Path("/nonexistent"))
    _inject_stats(model, cfg)
    model.eval()
    dummy = torch.randn(2, cfg["sample_rate"])
    with torch.no_grad():
        probs = model(dummy)
    assert probs.shape[0] == 2
    assert probs.shape[1] == cfg["model"]["n_classes"]
    assert 0.0 <= probs.min().item() <= 1.0
    assert 0.0 <= probs.max().item() <= 1.0


def test_own_model_custom_cfg_override_applies():
    """Passing a cfg with tail override should produce a tailed OwnModel."""
    cfg = get_own_config({"tail": "gru", "tail_width": 32})
    model = OwnModel(cfg=cfg, stats_path=Path("/nonexistent"))
    assert model.tail is not None
    _inject_stats(model, cfg)
    model.eval()
    dummy = torch.randn(1, cfg["sample_rate"])
    with torch.no_grad():
        probs = model(dummy)
    assert probs.shape[1] == cfg["model"]["n_classes"]
