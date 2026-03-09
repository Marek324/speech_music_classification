# tcn/evaluation.py
# Evaluation interface for TCN model. Handles data loading and inference.

import logging
import time

import numpy as np
import torch

from ..evaluator import EvalResultsBoth, run_evaluation
from ..wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log

from .config import get_config, get_weights_path
from .dataset import load_tcn_dataset
from .model import SpeechMusicDetector

log = logging.getLogger(__name__)


def get_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load TCN model and test data, run inference. Returns (y_true, y_pred, subclasses, time_per_sample_ns).
    Labels: -1=speech, 1=music. Uses full test dataset.
    """
    cfg = get_config()
    sr = cfg["sample_rate"]
    path = get_weights_path()

    if not path.exists():
        raise FileNotFoundError(f"TCN weights not found at {path}. Run train first.")

    ds_link = cfg["dataset"]["eval"]

    model = SpeechMusicDetector(sample_rate=sr)
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    model.load_state_dict(state)
    model.eval()

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    subclasses_list: list[str] = []

    t0 = time.perf_counter_ns()
    total_frames = 0
    for wav, targets, subclass in load_tcn_dataset(
        ds_link, "test", yield_subclass=True
    ):
        with torch.no_grad():
            probs = model(wav)  # (1, 2, T)
        pred = (probs[0, 1, :] > 0.5).float()  # (T,) music prob per frame
        target_class = int(targets[1, 0].item())  # 0 or 1 (same for all frames)
        y_true_frame = -1 if target_class == 0 else 1
        pred_frame = pred.cpu().numpy()
        T = pred_frame.shape[0]
        y_true_list.extend([y_true_frame] * T)
        y_pred_list.extend([-1 if p <= 0.5 else 1 for p in pred_frame])
        subclasses_list.extend([subclass] * T)
        total_frames += T
    classify_ns = time.perf_counter_ns() - t0

    time_per_sample_ns = classify_ns / total_frames if total_frames else 0.0

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    y_sub = np.array(subclasses_list, dtype="U40")

    return y_true, y_pred, y_sub, time_per_sample_ns


def eval_tcn(save_to_file: bool = True) -> EvalResultsBoth:
    """Evaluate TCN model on full test split. Uses main evaluator for metrics and output."""
    y_true, y_pred, subclasses, time_per_sample_ns = get_predictions()
    both = run_evaluation(
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
        "eval/accuracy": both.two_class.accuracy,
        "eval/f1": both.two_class.f1,
        "eval/correct": int(both.two_class.accuracy * n),
        "eval/total": n,
    })
    wandb_finish()

    return both
