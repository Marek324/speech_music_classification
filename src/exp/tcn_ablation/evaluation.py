# exp/tcn_ablation/evaluation.py

import logging

import torch

from ...evaluator import EvalResults, run_evaluation
from ...wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log
from ...nn.evaluation import run_nn_inference
from .config import get_config, get_weights_path, get_preprocess_stats_path
from .model import ExpSpeechMusicDetector

log = logging.getLogger(__name__)


def _load_exp_model(cfg: dict) -> ExpSpeechMusicDetector:
    name = cfg["name"]
    weights_path = get_weights_path(name)
    if not weights_path.exists():
        raise FileNotFoundError(f"Experiment weights not found at {weights_path}. Run train first.")

    stats_path = get_preprocess_stats_path(name)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Preprocess stats not found at {stats_path}. Run train first to compute them."
        )

    model = ExpSpeechMusicDetector(cfg)
    from safetensors.torch import load_file
    state = load_file(weights_path, device="cpu")
    model.load_state_dict(state)
    return model


def eval_exp(save_to_file: bool = True) -> EvalResults:
    """Evaluate experiment on the test split. Reads config from config.toml."""
    cfg = get_config()
    name = cfg["name"]
    model = _load_exp_model(cfg)

    y_true, y_pred, subclasses, time_per_sample_ns = run_nn_inference(
        model, cfg["dataset"]["eval"], cfg
    )
    res = run_evaluation(
        y_true,
        y_pred,
        subclasses,
        time_per_sample_ns,
        output_name=f"exp_tcn_ablation_{name}",
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
