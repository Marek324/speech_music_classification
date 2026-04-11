# tcn/evaluation.py
# Evaluation interface for TCN model.

import logging
from pathlib import Path

import torch

from ...evaluator import EvalResults, run_evaluation
from ...wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log

from ..evaluation import run_nn_inference
from .config import get_config, get_weights_path, get_preprocess_stats_path
from .model import SpeechMusicDetector
from .preprocess import validate_preprocess_stats

log = logging.getLogger(__name__)


def _load_tcn_model(weights_path=None, cfg: dict | None = None, stats_path=None) -> SpeechMusicDetector:
    """Load TCN model from weights file."""
    if cfg is None:
        cfg = get_config()
    path = Path(weights_path) if weights_path is not None else get_weights_path()

    if not path.exists():
        raise FileNotFoundError(f"TCN weights not found at {path}. Run train first.")

    if stats_path is None:
        validate_preprocess_stats()
    else:
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Preprocess stats not found at {stats_path}. Run train first.")

    model = SpeechMusicDetector(cfg=cfg, stats_path=stats_path)
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    model.load_state_dict(state)
    return model


def eval_tcn(
    save_to_file: bool = True,
    weights_path=None,
    cfg: dict | None = None,
    stats_path=None,
    output_name: str | None = None,
    output_dir: Path | None = None,
) -> EvalResults:
    """Evaluate TCN model on full test split. Uses main evaluator for metrics and output.

    *output_dir* is forwarded to ``run_evaluation()`` to route the saved ``.eval`` file
    into an experiment-specific directory instead of the default ``repo_root/results/``.
    """
    if cfg is None:
        cfg = get_config()
    model = _load_tcn_model(weights_path=weights_path, cfg=cfg, stats_path=stats_path)

    y_true, y_pred, subclasses, time_per_sample_ns, device = run_nn_inference(
        model, cfg["dataset"], cfg
    )
    res = run_evaluation(
        y_true,
        y_pred,
        subclasses,
        time_per_sample_ns,
        output_name=output_name or "tcn",
        save_to_file=save_to_file,
        device=device,
        output_dir=output_dir,
    )

    n = len(y_true)
    wandb_init(config={})
    wandb_log({
        "eval/accuracy": res.accuracy,
        "eval/f1": res.f1,
        "eval/correct": int(res.accuracy * n),
        "eval/total": n,
    })
    wandb_finish()

    return res
