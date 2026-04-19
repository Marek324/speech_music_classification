# tests/test_nn_tcn_config.py

import pytest
from pathlib import Path
from src.nn.tcn.config import (
    extract_overrides,
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
# Tests reference only variants that are currently active in
# src/exp/tcn_ablation/config.toml. Commented-out ablations (capacity,
# stacks, kernel, training/seq_len, n_mels, batch_size, augmentation, loss)
# were removed per the latest ablation notes — don't resurrect tests for them.

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
    cfg = get_ablation_config("layers_2", config_path=ABLATION_CFG, subgroup="layers")
    assert cfg["model"]["n_layers"] == 2


def test_get_ablation_config_dropout_low():
    cfg = get_ablation_config("dropout_low", config_path=ABLATION_CFG, subgroup="regularization")
    assert cfg["model"]["dropout"] == pytest.approx(0.1)


def test_get_ablation_config_baseline_equals_base():
    base = get_config(ABLATION_CFG)
    for subgroup in list_ablations(ABLATION_CFG):
        abl = get_ablation_config("baseline", config_path=ABLATION_CFG, subgroup=subgroup)
        assert abl["model"] == base["model"], f"baseline mismatch in subgroup {subgroup}"


def test_get_ablation_config_non_overridden_keys_preserved():
    base = get_config(ABLATION_CFG)
    cfg = get_ablation_config("layers_2", config_path=ABLATION_CFG, subgroup="layers")
    # keys other than n_layers should be unchanged
    for k in ("kernel_size", "n_filters", "n_stacks", "dropout"):
        assert cfg["model"][k] == base["model"][k]


def test_get_ablation_config_sets_name():
    cfg = get_ablation_config("layers_2", config_path=ABLATION_CFG, subgroup="layers")
    assert cfg["name"] == "layers_2"


# ── list_ablations ────────────────────────────────────────────────────────

def test_list_ablations_returns_dict():
    result = list_ablations(ABLATION_CFG)
    assert isinstance(result, dict)


def test_list_ablations_subgroups_present():
    result = list_ablations(ABLATION_CFG)
    # Only currently-active subgroups — see ablation notes for which have been dropped.
    for sg in ("optimizer", "layers", "regularization", "skip_connections", "activation"):
        assert sg in result


def test_list_ablations_layers_variants():
    result = list_ablations(ABLATION_CFG)
    assert set(result["layers"]) == {"baseline", "layers_1", "layers_2", "layers_3"}


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
    assert p.parent.name == "weights"


def test_get_weights_path_named():
    p = get_weights_path("foo")
    assert p.name == "tcn_foo.safetensors"
    assert p.parent.name == "weights"


def test_get_weights_path_with_subdir():
    """subdir scopes weights into a per-experiment directory; collision-proof across experiments."""
    p = get_weights_path("baseline", subdir="tcn_hybrid")
    assert p.name == "tcn_baseline.safetensors"
    assert p.parent.name == "tcn_hybrid"
    assert p.parent.parent.name == "weights"


def test_get_weights_path_subdir_no_name():
    p = get_weights_path(subdir="tcn_hybrid")
    assert p.name == "tcn.safetensors"
    assert p.parent.name == "tcn_hybrid"


def test_get_preprocess_stats_path_default():
    p = get_preprocess_stats_path()
    assert "preprocess_stats" in p.name
    assert p.parent.name == "weights"


def test_get_preprocess_stats_path_named():
    p = get_preprocess_stats_path(name="foo")
    assert p.name == "tcn_foo_preprocess_stats.pt"
    assert p.parent.name == "weights"


def test_get_preprocess_stats_path_with_subdir():
    """Same subdir scoping as weights — stats and weights live side-by-side."""
    p = get_preprocess_stats_path(name="baseline", subdir="tcn_hybrid")
    assert p.name == "tcn_baseline_preprocess_stats.pt"
    assert p.parent.name == "tcn_hybrid"
    assert p.parent.parent.name == "weights"


def test_paths_are_collision_free_across_experiments():
    """Two experiments with the same variant name must produce different paths."""
    a = get_weights_path("baseline", subdir="tcn_hybrid")
    b = get_weights_path("baseline", subdir="tcn_combined")
    assert a != b
    assert a.parent != b.parent
    sa = get_preprocess_stats_path(name="baseline", subdir="tcn_hybrid")
    sb = get_preprocess_stats_path(name="baseline", subdir="tcn_combined")
    assert sa != sb


# ── New config keys: batch_size, augment, activation ─────────────────────────

def test_get_config_has_batch_size():
    cfg = get_config(ABLATION_CFG)
    assert "batch_size" in cfg
    assert isinstance(cfg["batch_size"], int)


def test_get_config_has_augment():
    cfg = get_config(ABLATION_CFG)
    assert "augment" in cfg
    assert isinstance(cfg["augment"], bool)


def test_get_config_model_has_activation():
    cfg = get_config(ABLATION_CFG)
    assert "activation" in cfg["model"]
    assert cfg["model"]["activation"] == "relu"


def test_ablation_activation_leaky_relu():
    cfg = get_ablation_config("leaky_relu", config_path=ABLATION_CFG, subgroup="activation")
    assert cfg["model"]["activation"] == "leaky_relu"


def test_ablation_activation_gelu():
    cfg = get_ablation_config("gelu", config_path=ABLATION_CFG, subgroup="activation")
    assert cfg["model"]["activation"] == "gelu"


def test_ablation_skip_connections_no_skip():
    cfg = get_ablation_config("no_skip", config_path=ABLATION_CFG, subgroup="skip_connections")
    assert cfg["model"]["skip_connections"] is False


# ── carried_overrides ─────────────────────────────────────────────────────────

def test_carried_overrides_top_key():
    carried = {"optimizer": "adam", "lr": 1e-3}
    cfg = get_ablation_config("baseline", config_path=ABLATION_CFG, subgroup="layers",
                              carried_overrides=carried)
    assert cfg["optimizer"] == "adam"
    assert cfg["lr"] == pytest.approx(1e-3)


def test_carried_overrides_model_key():
    carried = {"n_filters": 32}
    cfg = get_ablation_config("baseline", config_path=ABLATION_CFG, subgroup="layers",
                              carried_overrides=carried)
    assert cfg["model"]["n_filters"] == 32


def test_carried_overrides_variant_wins_over_carried():
    """Variant-specific overrides take precedence over carried config."""
    carried = {"n_layers": 4}
    cfg = get_ablation_config("layers_2", config_path=ABLATION_CFG, subgroup="layers",
                              carried_overrides=carried)
    assert cfg["model"]["n_layers"] == 2


def test_carried_overrides_none_unchanged():
    base = get_ablation_config("baseline", config_path=ABLATION_CFG, subgroup="layers")
    carried_none = get_ablation_config("baseline", config_path=ABLATION_CFG, subgroup="layers",
                                       carried_overrides=None)
    assert base["model"] == carried_none["model"]


# ── extract_overrides ─────────────────────────────────────────────────────────

def test_extract_overrides_returns_flat_dict():
    cfg = get_config(ABLATION_CFG)
    out = extract_overrides(cfg)
    assert isinstance(out, dict)
    # Must not contain nested dicts
    assert not any(isinstance(v, dict) for v in out.values())


def test_extract_overrides_contains_top_keys():
    cfg = get_config(ABLATION_CFG)
    out = extract_overrides(cfg)
    for k in ("optimizer", "lr", "seq_len", "loss"):
        assert k in out, f"missing key: {k}"


def test_extract_overrides_contains_model_keys():
    cfg = get_config(ABLATION_CFG)
    out = extract_overrides(cfg)
    for k in ("n_filters", "dropout", "activation"):
        assert k in out, f"missing model key: {k}"


def test_extract_overrides_roundtrip():
    """Overrides extracted from a variant config should reproduce that config when carried."""
    cfg1 = get_ablation_config("adam", config_path=ABLATION_CFG, subgroup="optimizer")
    overrides = extract_overrides(cfg1)
    cfg2 = get_ablation_config("baseline", config_path=ABLATION_CFG, subgroup="layers",
                               carried_overrides=overrides)
    assert cfg2["optimizer"] == "adam"
    assert cfg2["lr"] == pytest.approx(1e-3)


# ── Subgroups present in list_ablations ───────────────────────────────────────
# Historical note: n_mels / batch_size / augmentation / capacity / training /
# stacks / kernel / loss were commented out of tcn_ablation/config.toml after
# their respective signals turned out to be flat. Tests for those live only
# in the git history.

def test_list_ablations_active_subgroups():
    result = list_ablations(ABLATION_CFG)
    # These are the subgroups still active in the current ablation config.
    for sg in ("optimizer", "layers", "regularization", "skip_connections", "activation"):
        assert sg in result, f"subgroup '{sg}' missing from list_ablations"


def test_list_ablations_activation_variants():
    result = list_ablations(ABLATION_CFG)
    assert set(result["activation"]) == {"baseline", "leaky_relu", "elu", "gelu"}
