# tests/test_exp_subdir_isolation.py
# Sanity check: every experiment scopes its weights and preprocess-stats under
# weights/<exp_name>/. Two experiments using the same variant name (e.g. "baseline")
# must NOT collide on disk — that bug already cost us once when log_mel stats
# leaked from tcn_combined into tcn_hybrid (240 vs 80 channel mismatch).

import importlib

import pytest

# (cli_module_path, expected_subdir)
_EXPERIMENTS = [
    ("src.exp.tcn_hybrid.cli",     "tcn_hybrid"),
    ("src.exp.tcn_combined.cli",   "tcn_combined"),
    ("src.exp.tcn_frontend.cli",   "tcn_frontend"),
    ("src.exp.nn_architecture.cli","nn_architecture"),
    ("src.exp.nn_preprocessor.cli","nn_preprocessor"),
    ("src.exp.tcn_ablation.cli",   "tcn_ablation"),
]


@pytest.mark.parametrize("module_path,expected_subdir", _EXPERIMENTS)
def test_exp_module_declares_subdir(module_path, expected_subdir):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "_EXP_SUBDIR"), (
        f"{module_path} must define _EXP_SUBDIR to scope its weights"
    )
    assert mod._EXP_SUBDIR == expected_subdir, (
        f"{module_path}._EXP_SUBDIR = {mod._EXP_SUBDIR!r}, expected {expected_subdir!r}"
    )


@pytest.mark.parametrize("module_path,expected_subdir", _EXPERIMENTS[:5])  # ablation _load needs subgroup
def test_variant_load_uses_subdir(module_path, expected_subdir):
    """_load returns paths under weights/<expected_subdir>/."""
    mod = importlib.import_module(module_path)
    cfg, name, weights_path, stats_path = mod._load("baseline")
    assert weights_path.parent.name == expected_subdir, (
        f"{module_path}: weights_path={weights_path}"
    )
    assert stats_path.parent.name == expected_subdir, (
        f"{module_path}: stats_path={stats_path}"
    )


def test_tcn_ablation_load_uses_subdir():
    """tcn_ablation._load takes subgroup; verify subdir scoping for a real subgroup variant."""
    from src.exp.tcn_ablation.cli import _load
    cfg, name, weights_path, stats_path = _load("adam", "optimizer")
    assert weights_path.parent.name == "tcn_ablation"
    assert stats_path.parent.name == "tcn_ablation"


def test_two_experiments_baseline_paths_differ():
    """Cross-experiment collision check: tcn_hybrid baseline and tcn_combined baseline
    must NOT share a file (the bug that motivated this refactor)."""
    from src.exp.tcn_hybrid.cli import _load as load_hybrid
    from src.exp.tcn_combined.cli import _load as load_combined
    _, _, wh, sh = load_hybrid("baseline")
    _, _, wc, sc = load_combined("baseline")
    assert wh != wc, "hybrid baseline weights collide with combined baseline weights"
    assert sh != sc, "hybrid baseline stats collide with combined baseline stats"
