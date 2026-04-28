# config.py
# Marek Hric

import copy
import tomli
from pathlib import Path
from typing import Any, Dict, Optional

Config = Dict[str, Any]

_cfg: Optional[Config] = None


def init_config(
    config_file_path: Path | None = None,
    model_name: str = "decision_tree",
    dataset_override: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load config and merge model-specific settings.
    Pass model_name to select buffers/settings for that model.

    *dataset_override*: when provided, replaces the loaded ``[dataset]`` section
    (used by experiment harnesses that point eval at a non-default tier — e.g.
    ``src/exp/critical/`` running against a local crit-tier parquet directory).
    """
    global _cfg
    if _cfg is not None:
        return _cfg

    if config_file_path is None:
        config_file_path = Path(__file__).parent.parent / "config.toml"

    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_file_path}")

    with open(config_file_path, "rb") as f:
        raw = tomli.load(f)

    buffers_by_model = raw.get("buffers", {})
    if model_name not in buffers_by_model:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(buffers_by_model.keys())}"
        )

    _cfg = {
        "sample_rate": raw["sample_rate"],
        "channels": raw["channels"],
        "n_fft": raw["n_fft"],
        "model": {"name": model_name},
        "buffers": copy.deepcopy(buffers_by_model[model_name]),
        "features": copy.deepcopy(raw.get("features", {})),
    }
    if dataset_override is not None:
        _cfg["dataset"] = copy.deepcopy(dataset_override)
    elif "dataset" in raw:
        _cfg["dataset"] = raw["dataset"]

    return _cfg


def reset_config() -> None:
    """Clear the singleton — required when re-initialising with a different override."""
    global _cfg
    _cfg = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        raise RuntimeError("Config not initialized")

    return _cfg
