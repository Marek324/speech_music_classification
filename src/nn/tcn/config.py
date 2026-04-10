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
    "optimizer": "adam",
    "lr": 1e-3,
    "seq_len": 128,
    "model": {
        "n_filters": 32,
        "kernel_size": 5,
        "n_layers": 6,
        "n_stacks": 2,
        "dropout": 0.2,
        "n_classes": 3,
        "use_weight_norm": False,
    },
}


def get_weights_path(name: str | None = None) -> Path:
    """Path to saved TCN weights (safetensors format).

    When *name* is provided, returns a per-experiment path
    ``weights/tcn_{name}.safetensors`` so ablation runs don't overwrite the
    production checkpoint.
    """
    weights = Path(__file__).resolve().parent.parent.parent.parent / "weights"
    if name:
        return weights / f"tcn_{name}.safetensors"
    return weights / "tcn.safetensors"


def get_preprocess_stats_path(revision: str | None = None, name: str | None = None) -> Path:
    """Path to precomputed normalization stats (mean, std) for log-mel spectrograms.

    *name* is checked first (per-experiment path); *revision* is the legacy
    dataset-revision suffix used by the production TCN.
    """
    weights = Path(__file__).resolve().parent.parent.parent.parent / "weights"
    if name:
        return weights / f"tcn_{name}_preprocess_stats.pt"
    if revision:
        return weights / f"tcn_preprocess_stats_{revision}.pt"
    return weights / "tcn_preprocess_stats.pt"


_MODEL_KEYS = frozenset([
    "n_filters", "kernel_size", "n_layers", "n_stacks",
    "dropout", "n_classes", "use_weight_norm", "skip_connections",
])
_TOP_KEYS = frozenset([
    "sample_rate", "n_fft", "hop_length", "n_mels", "f_min", "f_max",
    "optimizer", "lr", "seq_len",
])
_VARIANT_OVERRIDE_KEYS = _TOP_KEYS | _MODEL_KEYS | frozenset(["name"])


def _is_subgroup(val: dict) -> bool:
    """True if val looks like a subgroup dict (keys are variant names, not override keys)."""
    if not val:
        return False
    return not any(k in _VARIANT_OVERRIDE_KEYS for k in val)


def get_ablation_config(
    name: str,
    config_path: Path | None = None,
    subgroup: str | None = None,
) -> Dict[str, Any]:
    """Base [tcn] config merged with [tcn.ablations[.<subgroup>].<name>] overrides."""
    cfg = get_config(config_path)
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        raw = tomli.load(f)
    ablations = raw.get("tcn", {}).get("ablations", {})
    source = ablations.get(subgroup, {}) if subgroup else ablations
    overrides = source.get(name, {})
    for k in _TOP_KEYS:
        if k in overrides:
            cfg[k] = overrides[k]
    cfg["model"] = {**cfg["model"], **{k: overrides[k] for k in _MODEL_KEYS if k in overrides}}
    cfg["name"] = name
    return cfg


def list_ablations(config_path: Path | None = None) -> Dict[str, list]:
    """Return {subgroup: [name, ...]} for grouped ablations, or {"": [name, ...]} for flat."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        raw = tomli.load(f)
    ablations = raw.get("tcn", {}).get("ablations", {})
    result: Dict[str, list] = {}
    for key, val in ablations.items():
        if _is_subgroup(val):
            result[key] = list(val.keys())
        else:
            result.setdefault("", []).append(key)
    return result


def get_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load TCN config from config.toml [tcn] and [dataset] sections."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "config.toml"

    cfg = dict(_DEFAULT)
    cfg["dataset"] = {"url": None, "name": "full"}

    if not config_path.exists():
        return cfg

    with open(config_path, "rb") as f:
        raw = tomli.load(f)

    tcn = raw.get("tcn", {})
    for k in ("sample_rate", "n_fft", "hop_length", "n_mels", "f_min", "f_max", "optimizer", "lr", "seq_len", "name"):
        if k in tcn:
            cfg[k] = tcn[k]
    cfg["model"] = {**_DEFAULT["model"], **raw.get("tcn", {}).get("model", {})}
    if "dataset" in raw:
        cfg["dataset"] = {**cfg["dataset"], **raw["dataset"]}
    return cfg
