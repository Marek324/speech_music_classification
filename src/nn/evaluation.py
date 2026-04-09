# nn/evaluation.py
# Shared inference loop for NN models producing (1, 3, T) per-frame probabilities.

import logging
import time

import numpy as np
import torch

from .dataset import load_nn_dataset
from ..common import LABEL_MAP
from ..evaluator import _device_label

log = logging.getLogger(__name__)


def run_nn_inference(
    model,
    eval_cfg: dict,
    cfg: dict,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Run inference for any model producing (1, 3, T) probs per clip.

    Args:
        model: loaded PyTorch model (eval mode will be set internally)
        eval_cfg: dict with keys "url" and "name" (dataset config)
        cfg: dict with keys "sample_rate", "hop_length", "n_fft"
        max_rows: optional limit on test rows

    Returns:
        (y_true, y_pred, subclasses, time_per_sample_ns)
        Labels: -1=speech, 1=music, 2=inactive
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sr = cfg["sample_rate"]
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    subclasses_list: list[str] = []

    classify_ns = 0
    total_frames = 0
    for wav, targets, cls_name, subclass in load_nn_dataset(
        eval_cfg["url"], "test", sr, hop, n_fft,
        max_rows=max_rows, yield_subclass=True, name=eval_cfg["name"],
    ):
        wav_gpu = wav.to(device)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            probs = model(wav_gpu)  # (1, 3, T) — includes mel preprocessing
        classify_ns += time.perf_counter_ns() - t0
        T = probs.shape[-1]
        T_actual = min(T, targets.shape[-1])  # may be less than T due to center-padding in STFT

        # Per-frame y_pred: argmax across (speech, music, inactive) channels
        class_probs = probs[0, :, :T_actual]          # (3, T)
        pred_idx = class_probs.argmax(dim=0)           # 0=speech, 1=music, 2=inactive
        label_map = torch.tensor([-1, 1, 2], device=pred_idx.device)
        y_pred_frames = label_map[pred_idx].cpu().numpy()

        # Per-frame y_true: clip-uniform, sourced from the dataset `class` column.
        clip_label = LABEL_MAP[cls_name]
        y_true_frames = np.full(T_actual, clip_label, dtype=np.int64)

        y_true_list.extend(y_true_frames.tolist())
        y_pred_list.extend(y_pred_frames.tolist())
        subclasses_list.extend([subclass] * T_actual)
        total_frames += T_actual

    time_per_sample_ns = classify_ns / total_frames if total_frames else 0.0
    device_str = _device_label(device.type == "cuda")

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    y_sub = np.array(subclasses_list, dtype="U40")

    return y_true, y_pred, y_sub, time_per_sample_ns, device_str
