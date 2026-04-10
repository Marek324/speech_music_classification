# tests/test_nn_tcn_config.py

import pytest
from pathlib import Path
from src.nn.tcn.config import (
    get_config,
    get_ablation_config,
    list_ablations,
    get_weights_path,
    get_preprocess_stats_path,
    _is_subgroup,
    _DEFAULT,
)

ABLATION_CFG = Path(__file__).resolve().parent.parent / "src" / "exp" / "tcn_ablation" / "config.toml"
ROOT_CFG = Path(__file__).resolve().parent.parent / "config.toml"


# ── get_config ────────────────────────────────────────────────────────────

def test_get_config_returns_dict():
    cfg = get_config(ROOT_CFG)
    assert isinstance(cfg, dict)


def test_get_config_has_model_key():
    cfg = get_config(ROOT_CFG)
    assert "model" in cfg


def test_get_config_model_has_required_keys():
    cfg = get_config(ROOT_CFG)
    for k in ("n_filters", "kernel_size", "n_layers", "n_stacks", "dropout", "n_classes"):
        assert k in cfg["model"], f"missing model key: {k}"


def test_get_config_defaults_when_no_file(tmp_path):
    missing = tmp_path / "nonexistent.toml"
    cfg = get_config(missing)
    assert cfg["model"]["n_filters"] == _DEFAULT["model"]["n_filters"]


def test_get_config_toml_overrides_scalar():
    cfg = get_config(ROOT_CFG)
    # Root config.toml sets sample_rate=16000 at top level (classic) but TCN overrides to 22050
    assert cfg["sample_rate"] == 22050


def test_get_config_model_overrides_from_toml():
    cfg = get_config(ROOT_CFG)
    assert cfg["model"]["n_filters"] == 16


# ── get_ablation_config ──────────────────────────────────────────────────

def test_get_ablation_config_model_key_n_filters():
    cfg = get_ablation_config("filters_8", config_path=ABLATION_CFG, subgroup="capacity")
    assert cfg["model"]["n_filters"] == 8


def test_get_ablation_config_model_key_n_filters_32():
    cfg = get_ablation_config("filters_32", config_path=ABLATION_CFG, subgroup="capacity")
    assert cfg["model"]["n_filters"] == 32


def test_get_ablation_config_top_key_optimizer():
    cfg = get_ablation_config("adam", config_path=ABLATION_CFG, subgroup="optimizer")
    assert cfg["optimizer"] == "adam"
    assert cfg["lr"] == pytest.approx(1e-3)


def test_get_ablation_config_model_key_use_weight_norm():
    cfg = get_ablation_config("batch_norm", config_path=ABLATION_CFG, subgroup="optimizer")
    assert cfg["model"]["use_weight_norm"] is False


def test_get_ablation_config_mixed_top_and_model():
    cfg = get_ablation_config("adam_batchnorm", config_path=ABLATION_CFG, subgroup="optimizer")
    assert cfg["optimizer"] == "adam"
    assert cfg["model"]["use_weight_norm"] is False


def test_get_ablation_config_layers_2():
    cfg = get_ablation_config("layers_2", config_path=ABLATION_CFG, subgroup="depth")
    assert cfg["model"]["n_layers"] == 2


def test_get_ablation_config_stacks_5():
    cfg = get_ablation_config("stacks_5", config_path=ABLATION_CFG, subgroup="depth")
    assert cfg["model"]["n_stacks"] == 5


def test_get_ablation_config_dropout_low():
    cfg = get_ablation_config("dropout_low", config_path=ABLATION_CFG, subgroup="regularization")
    assert cfg["model"]["dropout"] == pytest.approx(0.1)


def test_get_ablation_config_seq_len():
    cfg = get_ablation_config("seq_len_256", config_path=ABLATION_CFG, subgroup="training")
    assert cfg["seq_len"] == 256


def test_get_ablation_config_baseline_equals_base():
    base = get_config(ABLATION_CFG)
    for subgroup in list_ablations(ABLATION_CFG):
        abl = get_ablation_config("baseline", config_path=ABLATION_CFG, subgroup=subgroup)
        assert abl["model"] == base["model"], f"baseline mismatch in subgroup {subgroup}"


def test_get_ablation_config_non_overridden_keys_preserved():
    base = get_config(ABLATION_CFG)
    cfg = get_ablation_config("filters_8", config_path=ABLATION_CFG, subgroup="capacity")
    # keys other than n_filters should be unchanged
    for k in ("kernel_size", "n_layers", "n_stacks", "dropout"):
        assert cfg["model"][k] == base["model"][k]


def test_get_ablation_config_sets_name():
    cfg = get_ablation_config("filters_8", config_path=ABLATION_CFG, subgroup="capacity")
    assert cfg["name"] == "filters_8"


# ── list_ablations ────────────────────────────────────────────────────────

def test_list_ablations_returns_dict():
    result = list_ablations(ABLATION_CFG)
    assert isinstance(result, dict)


def test_list_ablations_subgroups_present():
    result = list_ablations(ABLATION_CFG)
    for sg in ("optimizer", "capacity", "depth", "regularization", "training"):
        assert sg in result


def test_list_ablations_capacity_variants():
    result = list_ablations(ABLATION_CFG)
    assert set(result["capacity"]) == {"baseline", "filters_8", "filters_32"}


def test_list_ablations_all_variants_are_lists():
    result = list_ablations(ABLATION_CFG)
    for sg, variants in result.items():
        assert isinstance(variants, list)
        assert len(variants) > 0


# ── _is_subgroup ──────────────────────────────────────────────────────────

def test_is_subgroup_empty_dict_returns_false():
    assert _is_subgroup({}) is False


def test_is_subgroup_override_keys_returns_false():
    assert _is_subgroup({"optimizer": "adam"}) is False


def test_is_subgroup_variant_names_returns_true():
    assert _is_subgroup({"baseline": {}, "filters_8": {}}) is True


# ── path helpers ──────────────────────────────────────────────────────────

def test_get_weights_path_default_name():
    p = get_weights_path()
    assert p.name == "tcn.safetensors"


def test_get_weights_path_named():
    p = get_weights_path("foo")
    assert p.name == "tcn_foo.safetensors"


def test_get_preprocess_stats_path_default():
    p = get_preprocess_stats_path()
    assert "preprocess_stats" in p.name


def test_get_preprocess_stats_path_named():
    p = get_preprocess_stats_path(name="foo")
    assert p.name == "tcn_foo_preprocess_stats.pt"
