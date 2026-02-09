# config.py
# Marek Hric


import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

type Config = Dict[str, Any]

_cfg: Optional[Config] = None


def init_config(config_file_path: Path) -> Config:
    global _cfg
    if _cfg is not None:
        return _cfg

    if config_file_path is None:
        config_file_path = Path(__file__).parent.parent / "config.toml"

    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_file_path}")

    with open(config_file_path, "rb") as f:
        _cfg = tomllib.load(f)

    return _cfg


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        raise RuntimeError("Config not initialized")

    return _cfg
