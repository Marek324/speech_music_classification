# tests/test_exp_tcn_hybrid.py
# Sanity tests for the tcn_hybrid experiment's variant config loader + CLI wiring.

import pytest

from src.exp.tcn_hybrid.cli import _get_variant_config, _list_variants


EXPECTED_VARIANTS = {
    "baseline",
    "tcn_gru",
    "tcn_gru_wide",
    "tcn_lstm",
    "tcn_attn",
    "tcn_two_branch",
}


def test_list_variants_contains_expected_set():
    names = set(_list_variants())
    assert names == EXPECTED_VARIANTS, f"unexpected variants: {names}"


def test_baseline_has_no_tail():
    cfg = _get_variant_config("baseline")
    assert cfg["model"].get("tail") in (None, "", "none")
    # Base recipe inherits delta2 + conv1d.
    assert cfg["frontend"] == "log_mel_delta2"
    assert cfg["model"]["preprocessor"] == "conv1d"


@pytest.mark.parametrize("name,tail,width", [
    ("tcn_gru", "gru", 32),
    ("tcn_gru_wide", "gru", 64),
    ("tcn_lstm", "lstm", 32),
    ("tcn_attn", "attn", 32),
    ("tcn_two_branch", "two_branch", 16),
])
def test_variant_overrides_route_into_model_bucket(name, tail, width):
    cfg = _get_variant_config(name)
    assert cfg["model"]["tail"] == tail, f"{name}: tail={cfg['model'].get('tail')}"
    assert cfg["model"]["tail_width"] == width


def test_attn_variant_has_n_heads():
    cfg = _get_variant_config("tcn_attn")
    assert cfg["model"]["n_heads"] == 4


def test_all_variants_inherit_base_recipe():
    """Variants should not accidentally lose the base frontend/preprocessor/TCN params."""
    for name in EXPECTED_VARIANTS:
        cfg = _get_variant_config(name)
        assert cfg["frontend"] == "log_mel_delta2", f"{name}"
        assert cfg["model"]["preprocessor"] == "conv1d", f"{name}"
        assert cfg["model"]["n_filters"] == 16, f"{name}"
        assert cfg["model"]["n_layers"] == 4, f"{name}"
        assert cfg["model"]["use_weight_norm"] is True, f"{name}"
        assert cfg["optimizer"] == "sgd", f"{name}"


def test_variant_name_is_set_on_cfg():
    cfg = _get_variant_config("tcn_gru")
    assert cfg["name"] == "tcn_gru"


def test_model_keys_include_tail_fields():
    """Regression: tail / tail_width must be in _MODEL_KEYS so overrides route correctly."""
    from src.nn.tcn.config import _MODEL_KEYS
    assert "tail" in _MODEL_KEYS
    assert "tail_width" in _MODEL_KEYS
