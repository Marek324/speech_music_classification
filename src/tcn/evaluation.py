# tcn/evaluation.py
# Evaluation interface for TCN model. Handles data loading and inference.

import logging
import time

import numpy as np
import torch

from ..evaluator import EvalResults, run_evaluation
from ..wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log

from .config import get_config, get_weights_path
from .dataset import load_tcn_dataset
from .model import SpeechMusicDetector
from .preprocess import validate_preprocess_stats

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

    validate_preprocess_stats()

    ds_url = cfg["dataset"]["url"]
    ds_rev = cfg["dataset"].get("revision")

    model = SpeechMusicDetector(sample_rate=sr)
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    model.load_state_dict(state)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    subclasses_list: list[str] = []

    t0 = time.perf_counter_ns()
    total_frames = 0
    for wav, targets, subclass in load_tcn_dataset(
        ds_url, "test", yield_subclass=True, revision=ds_rev
    ):
        with torch.no_grad():
            wav = wav.to(device)
            probs = model(wav)  # (1, 2, T)
        T = probs.shape[-1]
        tgt = targets[:, :T]  # align to model output length
        T_actual = tgt.shape[-1]  # may be less than T due to center-padding in STFT

        # Per-frame y_true: -1=speech, 1=music, 2=inactive
        speech_mask = tgt[0, :] == 1.0
        music_mask = tgt[1, :] == 1.0
        y_true_frames = torch.where(music_mask, 1, torch.where(speech_mask, -1, 2)).cpu().numpy()

        # Per-frame y_pred: argmax across (speech, music, inactive) channels
        class_probs = probs[0, :, :T_actual]          # (3, T)
        pred_idx = class_probs.argmax(dim=0)           # 0=speech, 1=music, 2=inactive
        label_map = torch.tensor([-1, 1, 2], device=pred_idx.device)
        y_pred_frames = label_map[pred_idx].cpu().numpy()

        y_true_list.extend(y_true_frames.tolist())
        y_pred_list.extend(y_pred_frames.tolist())
        subclasses_list.extend([subclass] * T_actual)
        total_frames += T_actual
    classify_ns = time.perf_counter_ns() - t0

    time_per_sample_ns = classify_ns / total_frames if total_frames else 0.0

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    y_sub = np.array(subclasses_list, dtype="U40")

    return y_true, y_pred, y_sub, time_per_sample_ns


def eval_tcn(save_to_file: bool = True) -> EvalResults:
    """Evaluate TCN model on full test split. Uses main evaluator for metrics and output."""
    y_true, y_pred, subclasses, time_per_sample_ns = get_predictions()
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
