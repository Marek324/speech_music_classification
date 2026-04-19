# nn/small_tcn/config.py
# Config loader for SmallTCN. Mirrors tcn_lstm/config.py — loads src/nn/small_tcn/config.toml
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


def _load_small_tcn_default() -> Dict[str, Any]:
    """Flatten small_tcn/config.toml into the {top-level keys, 'model': {...}, 'dataset': {...}} shape
    that the rest of the pipeline expects."""
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomli.load(f)
    tcn = raw.get("tcn", {})
    cfg: Dict[str, Any] = {k: v for k, v in tcn.items() if k != "model"}
    cfg["model"] = dict(tcn.get("model", {}))
    cfg["dataset"] = dict(raw.get("dataset", {}))
    return cfg


_SMALL_TCN_DEFAULT: Dict[str, Any] = _load_small_tcn_default()


def get_small_tcn_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return the small_tcn/ config, optionally applying flat ``overrides``.

    Overrides are a flat dict mixing top-level and model keys; routing follows
    the same ``_TOP_KEYS`` / ``_MODEL_KEYS`` split as the TCN ablation loader.
    """
    cfg = deepcopy(_SMALL_TCN_DEFAULT)
    if overrides:
        for k in _TOP_KEYS:
            if k in overrides:
                cfg[k] = overrides[k]
        cfg["model"] = {
            **cfg["model"],
            **{k: overrides[k] for k in _MODEL_KEYS if k in overrides},
        }
    return cfg


_SMALL_TCN_SUBDIR = "small_tcn"


def get_small_tcn_weights_path() -> Path:
    """Path to small_tcn/ weights (safetensors format)."""
    return get_weights_path(name="small", subdir=_SMALL_TCN_SUBDIR)


def get_small_tcn_stats_path() -> Path:
    """Path to small_tcn/ preprocess stats."""
    return get_preprocess_stats_path(name="small", subdir=_SMALL_TCN_SUBDIR)
