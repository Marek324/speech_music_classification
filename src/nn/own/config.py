# nn/own/config.py
# Config for the OwnModel — baked to the combined-experiment winner (delta2_conv1d).
# Provides a programmatic override hook so future experiments can sweep variants
# against the own/ model without editing this file.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from ..tcn.config import (
    _MODEL_KEYS,
    _TOP_KEYS,
    get_preprocess_stats_path,
    get_weights_path,
)


_OWN_DEFAULT: Dict[str, Any] = {
    "name": "own",
    "frontend": "log_mel_delta2",
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 80,
    "n_mfcc": 20,
    "f_min": 27.5,
    "f_max": 8000.0,
    "optimizer": "sgd",
    "lr": 1e-3,
    "seq_len": 128,
    "batch_size": 32,
    "augment": True,
    "loss": "bce_with_logits",
    "model": {
        "backbone": "tcn",
        "preprocessor": "conv1d",
        "n_filters": 16,
        "kernel_size": 5,
        "n_layers": 4,
        "n_stacks": 3,
        "n_heads": 4,
        "dropout": 0.5,
        "n_classes": 3,
        "use_weight_norm": True,
        "activation": "relu",
    },
    "dataset": {
        "url": "Marek324/speech-music-classification",
        "name": "full",
    },
}


def get_own_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return the own/ config, optionally applying flat ``overrides``.

    Overrides are a flat dict mixing top-level and model keys; routing follows
    the same ``_TOP_KEYS`` / ``_MODEL_KEYS`` split as the TCN ablation loader.
    This hook exists so future experiments can sweep variants against own/
    without editing the defaults dict.
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
