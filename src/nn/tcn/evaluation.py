# tcn/evaluation.py
# Evaluation interface for TCN model.

import logging

import torch

from ...evaluator import EvalResults, run_evaluation
from ...wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log

from ..evaluation import run_nn_inference
from .config import get_config, get_weights_path
from .model import SpeechMusicDetector
from .preprocess import validate_preprocess_stats

log = logging.getLogger(__name__)


def _load_tcn_model(weights_path=None) -> SpeechMusicDetector:
    """Load TCN model from weights file."""
    cfg = get_config()
    path = weights_path if weights_path is not None else get_weights_path()

    if not path.exists():
        raise FileNotFoundError(f"TCN weights not found at {path}. Run train first.")

    validate_preprocess_stats()

    model = SpeechMusicDetector(sample_rate=cfg["sample_rate"])
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    model.load_state_dict(state)
    return model


def eval_tcn(save_to_file: bool = True, weights_path=None) -> EvalResults:
    """Evaluate TCN model on full test split. Uses main evaluator for metrics and output."""
    cfg = get_config()
    model = _load_tcn_model(weights_path=weights_path)

    y_true, y_pred, subclasses, time_per_sample_ns = run_nn_inference(
        model, cfg["dataset"]["eval"], cfg
    )
    res = run_evaluation(
        y_true,
        y_pred,
        subclasses,
        time_per_sample_ns,
        output_name="tcn",
        save_to_file=save_to_file,
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
