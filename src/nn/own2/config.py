# nn/own2/config.py
# Config loader for Own2Model. Mirrors own/config.py — loads src/nn/own2/config.toml
# and exposes the same overridable API as the TCN ablation loader.

import tomli
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from ..tcn.config import (
    _MODEL_KEYS,
    _TOP_KEYS,
    get_preprocess_stats_path,
    get_weights_path,
)


_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


def _load_own2_default() -> Dict[str, Any]:
    """Flatten own2/config.toml into the {top-level keys, 'model': {...}, 'dataset': {...}} shape
    that the rest of the pipeline expects."""
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomli.load(f)
    tcn = raw.get("tcn", {})
    cfg: Dict[str, Any] = {k: v for k, v in tcn.items() if k != "model"}
    cfg["model"] = dict(tcn.get("model", {}))
    cfg["dataset"] = dict(raw.get("dataset", {}))
    return cfg


_OWN2_DEFAULT: Dict[str, Any] = _load_own2_default()


def get_own2_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return the own2/ config, optionally applying flat ``overrides``.

    Overrides are a flat dict mixing top-level and model keys; routing follows
    the same ``_TOP_KEYS`` / ``_MODEL_KEYS`` split as the TCN ablation loader.
    """
    cfg = deepcopy(_OWN2_DEFAULT)
    if overrides:
        for k in _TOP_KEYS:
            if k in overrides:
                cfg[k] = overrides[k]
        cfg["model"] = {
            **cfg["model"],
            **{k: overrides[k] for k in _MODEL_KEYS if k in overrides},
        }
    return cfg


_OWN2_SUBDIR = "own2"


def get_own2_weights_path() -> Path:
    """Path to own2/ weights (safetensors format)."""
    return get_weights_path(name="own2", subdir=_OWN2_SUBDIR)


def get_own2_stats_path() -> Path:
    """Path to own2/ preprocess stats."""
    return get_preprocess_stats_path(name="own2", subdir=_OWN2_SUBDIR)
