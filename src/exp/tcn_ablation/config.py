# exp/tcn_ablation/config.py

import tomli
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # src/exp/tcn_ablation -> repo root

_DEFAULT: Dict[str, Any] = {
    "name": "baseline",
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 80,
    "f_min": 27.5,
    "f_max": 8000.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "seq_len": 128,
    "model": {
        "n_filters": 16,
        "kernel_size": 5,
        "n_layers": 4,
        "n_stacks": 3,
        "dropout": 0.5,
        "n_classes": 3,
        "use_weight_norm": False,
    },
}


def get_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load experiment config from src/exp/tcn_ablation/config.toml.
    Dataset settings are inherited from the root config.toml."""
    if config_path is None:
        config_path = _HERE / "config.toml"

    _empty_ds = {"url": None, "name": "mid"}
    cfg = dict(_DEFAULT)
    cfg["model"] = dict(_DEFAULT["model"])
    cfg["dataset"] = {
        "train": dict(_empty_ds),
        "eval": dict(_empty_ds),
        "stats": dict(_empty_ds),
    }

    if config_path.exists():
        with open(config_path, "rb") as f:
            raw = tomli.load(f)
        exp = raw.get("exp", {})
        for k in ("name", "sample_rate", "n_fft", "hop_length", "n_mels", "f_min", "f_max", "optimizer", "lr", "seq_len"):
            if k in exp:
                cfg[k] = exp[k]
        cfg["model"] = {**_DEFAULT["model"], **exp.get("model", {})}

    # Dataset URLs come from the root config.toml (shared with TCN)
    root_config = _ROOT / "config.toml"
    if root_config.exists():
        with open(root_config, "rb") as f:
            root_raw = tomli.load(f)
        if "dataset" in root_raw:
            cfg["dataset"] = {**cfg["dataset"], **root_raw["dataset"]}

    return cfg


def get_weights_path(name: str | None = None) -> Path:
    """Path to saved experiment weights."""
    cfg_name = name or get_config()["name"]
    return _ROOT / "weights" / f"exp_tcn_ablation_{cfg_name}.safetensors"


def get_preprocess_stats_path(name: str | None = None) -> Path:
    """Path to precomputed normalization stats for this experiment."""
    cfg_name = name or get_config()["name"]
    return _ROOT / "weights" / f"exp_tcn_ablation_{cfg_name}_preprocess_stats.pt"
