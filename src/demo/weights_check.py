# src/demo/weights_check.py
# Marek Hric

"""Cheap existence probe for model weights used by the demo UI.

Called from the page-load handler to badge the picker: no torch import here,
no model construction — just filesystem checks. Variant entries are derived
from src/nn/variants.toml so adding a variant requires no edits here.
"""

from pathlib import Path

import tomli

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WEIGHTS = _REPO_ROOT / "weights"
_VARIANTS_TOML = _REPO_ROOT / "src" / "nn" / "variants.toml"

CLASSIC_MODELS = ("decision_tree", "gmm", "svm")


def _variant_paths() -> dict[str, tuple[Path, Path]]:
    """Read variants.toml and derive (weights, stats) paths for each variant.

    Mirrors the path layout from `src/nn/tcn/config.py`:
        weights/{subdir}/tcn_{weights_stem}.safetensors
        weights/{subdir}/tcn_{weights_stem}_preprocess_stats.pt
    """
    if not _VARIANTS_TOML.exists():
        return {}
    with open(_VARIANTS_TOML, "rb") as f:
        data = tomli.load(f)
    out: dict[str, tuple[Path, Path]] = {}
    for name, spec in data.get("variants", {}).items():
        subdir = spec["subdir"]
        stem = spec["weights_stem"]
        out[name] = (
            _WEIGHTS / subdir / f"tcn_{stem}.safetensors",
            _WEIGHTS / subdir / f"tcn_{stem}_preprocess_stats.pt",
        )
    return out


# Canonical TCN lives at the root of weights/ (not in a subdir) and is not part
# of the variants registry — it's the paper baseline that variants extend.
_NN_PATHS: dict[str, tuple[Path, Path]] = {
    "tcn": (
        _WEIGHTS / "tcn.safetensors",
        _WEIGHTS / "tcn_preprocess_stats.pt",
    ),
    **_variant_paths(),
}

NN_MODELS: tuple[str, ...] = tuple(_NN_PATHS)
ALL_MODELS: tuple[str, ...] = CLASSIC_MODELS + NN_MODELS


def has_weights(model_name: str) -> bool:
    if model_name in CLASSIC_MODELS:
        return (_WEIGHTS / model_name).exists()
    if model_name in NN_MODELS:
        return all(p.exists() for p in _NN_PATHS[model_name])
    return False


def all_status() -> dict[str, bool]:
    return {name: has_weights(name) for name in ALL_MODELS}
