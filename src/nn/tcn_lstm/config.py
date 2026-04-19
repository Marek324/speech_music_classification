# nn/tcn_lstm/config.py
# Config loader for TCNLSTM. Reads src/nn/tcn_lstm/config.toml; exposes the same
# overridable API as the TCN ablation loader so future experiments can sweep
# variants against tcn_lstm/ without editing the TOML file.

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


def _load_tcn_lstm_default() -> Dict[str, Any]:
    """Flatten tcn_lstm/config.toml into the {top-level keys, 'model': {...}, 'dataset': {...}} shape
    that the rest of the pipeline expects."""
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomli.load(f)
    tcn = raw.get("tcn", {})
    cfg: Dict[str, Any] = {k: v for k, v in tcn.items() if k != "model"}
    cfg["model"] = dict(tcn.get("model", {}))
    cfg["dataset"] = dict(raw.get("dataset", {}))
    return cfg


_TCN_LSTM_DEFAULT: Dict[str, Any] = _load_tcn_lstm_default()


def get_tcn_lstm_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return the tcn_lstm/ config, optionally applying flat ``overrides``.

    Overrides are a flat dict mixing top-level and model keys; routing follows
    the same ``_TOP_KEYS`` / ``_MODEL_KEYS`` split as the TCN ablation loader.
    This hook exists so future experiments can sweep variants against tcn_lstm/
    without editing the defaults TOML.
    """
    cfg = deepcopy(_TCN_LSTM_DEFAULT)
    if overrides:
        for k in _TOP_KEYS:
            if k in overrides:
                cfg[k] = overrides[k]
        cfg["model"] = {
            **cfg["model"],
            **{k: overrides[k] for k in _MODEL_KEYS if k in overrides},
        }
    return cfg


_TCN_LSTM_SUBDIR = "tcn_lstm"


def get_tcn_lstm_weights_path() -> Path:
    """Path to tcn_lstm/ weights (safetensors format)."""
    return get_weights_path(name="lstm", subdir=_TCN_LSTM_SUBDIR)


def get_tcn_lstm_stats_path() -> Path:
    """Path to tcn_lstm/ preprocess stats."""
    return get_preprocess_stats_path(name="lstm", subdir=_TCN_LSTM_SUBDIR)
