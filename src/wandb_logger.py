# wandb_logger.py
# Marek Hric
# Optional Weights & Biases logging.

import logging
import os
from pathlib import Path
from typing import Any

import tomli

log = logging.getLogger(__name__)

_wandb = None


def _lazy_import():
    global _wandb
    if _wandb is None:
        try:
            import wandb as _w
            _wandb = _w
        except ImportError:
            _wandb = False
    return _wandb


def _load_wandb_config() -> dict:
    """Load [wandb] section from config.toml."""
    path = Path(__file__).resolve().parent.parent / "config.toml"
    if not path.exists():
        return {"project": "smclassifier", "entity": None}
    with open(path, "rb") as f:
        raw = tomli.load(f)
    return raw.get("wandb", {"project": "smclassifier", "entity": None})


def init(config: dict | None = None) -> bool:
    """Initialize wandb run. project/entity from config.toml [wandb].
    Skips init if WANDB_MODE=disabled. Gracefully degrades on network/permission errors."""
    if os.environ.get("WANDB_MODE") == "disabled":
        log.info("wandb disabled via WANDB_MODE=disabled")
        return False
    wb = _lazy_import()
    if wb is False:
        log.warning("wandb not installed; run pip install wandb")
        return False
    if wb.run is not None:
        return True
    wb_cfg = _load_wandb_config()
    kwargs = {"project": wb_cfg.get("project", "smclassifier"), "config": config or {}}
    if wb_cfg.get("entity"):
        kwargs["entity"] = wb_cfg["entity"]
    try:
        wb.init(**kwargs)
        return True
    except Exception as e:
        log.warning("wandb init failed (%s); continuing without wandb logging", e)
        return False


def log_metrics(metrics: dict[str, Any], step: int | None = None, commit: bool = True):
    """Log metrics to wandb."""
    wb = _lazy_import()
    if wb and wb.run is not None:
        wb.log(metrics, step=step, commit=commit)


def finish():
    """Finish the current wandb run."""
    wb = _lazy_import()
    if wb and wb.run is not None:
        wb.finish()


def is_active() -> bool:
    """Return True if wandb is initialized and running."""
    wb = _lazy_import()
    return bool(wb and wb.run is not None)
