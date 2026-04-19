# tests/test_nn_small_tcn.py
# Sanity tests for the small_tcn/ module: config, paths, model instantiation + forward.

from pathlib import Path

import pytest
import torch

from src.nn.small_tcn.config import (
    _SMALL_TCN_DEFAULT,
    get_small_tcn_config,
    get_small_tcn_stats_path,
    get_small_tcn_weights_path,
)
from src.nn.small_tcn.model import SmallTCN


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_small_tcn_default_is_delta2_n_filters_8():
    """The hardcoded default is the small-footprint variant."""
    assert _SMALL_TCN_DEFAULT["frontend"] == "log_mel_delta2"
    assert _SMALL_TCN_DEFAULT["model"]["preprocessor"] == "none"
    assert _SMALL_TCN_DEFAULT["model"]["backbone"] == "tcn"
    assert _SMALL_TCN_DEFAULT["model"]["n_filters"] == 8
    assert _SMALL_TCN_DEFAULT["model"]["n_layers"] == 4
    assert _SMALL_TCN_DEFAULT["model"]["n_stacks"] == 3
    assert "tail" not in _SMALL_TCN_DEFAULT["model"]
    assert _SMALL_TCN_DEFAULT["optimizer"] == "sgd"
    assert _SMALL_TCN_DEFAULT["name"] == "small"


def test_get_small_tcn_config_is_deep_copy():
    """Mutating the returned cfg must not affect subsequent calls."""
    cfg1 = get_small_tcn_config()
    cfg1["lr"] = 999.0
    cfg1["model"]["n_filters"] = 999
    cfg2 = get_small_tcn_config()
    assert cfg2["lr"] != 999.0
    assert cfg2["model"]["n_filters"] != 999


def test_get_small_tcn_config_no_overrides_returns_defaults():
    cfg = get_small_tcn_config()
    assert cfg["frontend"] == "log_mel_delta2"
    assert cfg["model"]["n_filters"] == 8


def test_get_small_tcn_config_applies_top_level_override():
    cfg = get_small_tcn_config({"lr": 5e-4, "optimizer": "adam"})
    assert cfg["lr"] == 5e-4
    assert cfg["optimizer"] == "adam"
    assert cfg["frontend"] == "log_mel_delta2"


def test_get_small_tcn_config_applies_model_override():
    cfg = get_small_tcn_config({"n_filters": 16, "dropout": 0.25})
    assert cfg["model"]["n_filters"] == 16
    assert cfg["model"]["dropout"] == 0.25
    assert cfg["model"]["n_layers"] == 4


def test_get_small_tcn_config_mixed_overrides():
    cfg = get_small_tcn_config({"lr": 1e-2, "n_filters": 16, "tail": "lstm", "tail_width": 32})
    assert cfg["lr"] == 1e-2
    assert cfg["model"]["n_filters"] == 16
    assert cfg["model"]["tail"] == "lstm"
    assert cfg["model"]["tail_width"] == 32


def test_get_small_tcn_config_ignores_unknown_keys():
    cfg = get_small_tcn_config({"nonsense_flag": 42})
    assert "nonsense_flag" not in cfg
    assert "nonsense_flag" not in cfg["model"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_small_tcn_weights_path_has_expected_suffix():
    p = get_small_tcn_weights_path()
    assert p.name == "tcn_small.safetensors"
    assert p.parent.name == "small_tcn"
    assert p.parent.parent.name == "weights"


def test_small_tcn_stats_path_has_expected_suffix():
    p = get_small_tcn_stats_path()
    assert p.name == "tcn_small_preprocess_stats.pt"
    assert p.parent.name == "small_tcn"
    assert p.parent.parent.name == "weights"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _inject_stats(model, cfg):
    model.fe.norm_mean = torch.zeros(1, model.fe.n_features, 1)
    model.fe.norm_std = torch.ones(1, model.fe.n_features, 1)
    model.fe._stats_loaded = True
    return model


def test_small_tcn_model_instantiates_with_defaults():
    model = SmallTCN(stats_path=Path("/nonexistent"))
    assert model.fe.n_features == 240  # log_mel_delta2 at n_mels=80 -> 3*80
    # No tail — minimal variant.
    assert model.tail is None


def test_small_tcn_model_forward_shape():
    cfg = get_small_tcn_config()
    model = SmallTCN(cfg=cfg, stats_path=Path("/nonexistent"))
    _inject_stats(model, cfg)
    model.eval()
    dummy = torch.randn(2, cfg["sample_rate"])
    with torch.no_grad():
        probs = model(dummy)
    assert probs.shape[0] == 2
    assert probs.shape[1] == cfg["model"]["n_classes"]
    assert 0.0 <= probs.min().item() <= 1.0
    assert 0.0 <= probs.max().item() <= 1.0
