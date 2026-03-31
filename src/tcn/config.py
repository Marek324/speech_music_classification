# tcn/config.py
# Marek Hric

import tomli
from pathlib import Path
from typing import Any, Dict

_DEFAULT: Dict[str, Any] = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 80,
    "f_min": 27.5,
    "f_max": 8000.0,
    "model": {
        "n_filters": 32,
        "kernel_size": 5,
        "n_layers": 6,
        "n_stacks": 2,
        "dropout": 0.2,
        "n_classes": 3,
    },
}


def get_weights_path() -> Path:
    """Path to saved TCN weights (safetensors format)."""
    return Path(__file__).resolve().parent.parent.parent / "weights" / "tcn.safetensors"


def get_preprocess_stats_path() -> Path:
    """Path to precomputed normalization stats (mean, std) for log-mel spectrograms."""
    return Path(__file__).resolve().parent.parent.parent / "weights" / "tcn_preprocess_stats.pt"


def get_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load TCN config from config.toml [tcn] and [dataset] sections."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config.toml"

    _empty_ds = {"url": None, "name": "full"}
    cfg = dict(_DEFAULT)
    cfg["dataset"] = {
        "train": dict(_empty_ds),
        "eval": dict(_empty_ds),
        "stats": dict(_empty_ds),
    }

    if not config_path.exists():
        return cfg

    with open(config_path, "rb") as f:
        raw = tomli.load(f)

    tcn = raw.get("tcn", {})
    for k in ("sample_rate", "n_fft", "hop_length", "n_mels", "f_min", "f_max"):
        if k in tcn:
            cfg[k] = tcn[k]
    cfg["model"] = {**_DEFAULT["model"], **raw.get("tcn", {}).get("model", {})}
    if "dataset" in raw:
        cfg["dataset"] = {**cfg["dataset"], **raw["dataset"]}
    return cfg
