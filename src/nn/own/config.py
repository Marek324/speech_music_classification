# nn/own/config.py
# Config loader for OwnModel. Reads src/nn/own/config.toml; exposes the same
# overridable API as the TCN ablation loader so future experiments can sweep
# variants against own/ without editing the TOML file.

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


def _load_own_default() -> Dict[str, Any]:
    """Flatten own/config.toml into the {top-level keys, 'model': {...}, 'dataset': {...}} shape
    that the rest of the pipeline expects."""
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomli.load(f)
    tcn = raw.get("tcn", {})
    cfg: Dict[str, Any] = {k: v for k, v in tcn.items() if k != "model"}
    cfg["model"] = dict(tcn.get("model", {}))
    cfg["dataset"] = dict(raw.get("dataset", {}))
    return cfg


_OWN_DEFAULT: Dict[str, Any] = _load_own_default()


def get_own_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return the own/ config, optionally applying flat ``overrides``.

    Overrides are a flat dict mixing top-level and model keys; routing follows
    the same ``_TOP_KEYS`` / ``_MODEL_KEYS`` split as the TCN ablation loader.
    This hook exists so future experiments can sweep variants against own/
    without editing the defaults TOML.
    """
    cfg = deepcopy(_OWN_DEFAULT)
    if overrides:
        for k in _TOP_KEYS:
            if k in overrides:
                cfg[k] = overrides[k]
        cfg["model"] = {
            **cfg["model"],
            **{k: overrides[k] for k in _MODEL_KEYS if k in overrides},
        }
    return cfg


_OWN_SUBDIR = "own"


def get_own_weights_path() -> Path:
    """Path to own/ weights (safetensors format)."""
    return get_weights_path(name="own", subdir=_OWN_SUBDIR)


def get_own_stats_path() -> Path:
    """Path to own/ preprocess stats."""
    return get_preprocess_stats_path(name="own", subdir=_OWN_SUBDIR)
