# src/nn/evaluation.py
# Marek Hric
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str, np.ndarray, np.ndarray]:
    """Run inference for any model producing (1, 3, T) probs per clip.

    Args:
        model: loaded PyTorch model (eval mode will be set internally)
        eval_cfg: dict with keys "url" and "name" (dataset config)
        cfg: dict with keys "sample_rate", "hop_length", "n_fft"
        max_rows: optional limit on test rows

    Returns:
        (y_true, y_pred, subclasses, time_per_frame_ns, device_str, y_scores, clip_ids)
        Labels: -1=speech, 1=music, 2=background
        y_scores: (N, 3) — columns [speech, music, background]
        clip_ids: (N,) int — per-frame source-clip index (0-indexed)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sr = cfg["sample_rate"]
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    scores_list: list[np.ndarray] = []
    subclasses_list: list[str] = []
    clip_ids_list: list[np.ndarray] = []

    classify_ns = 0
    total_frames = 0
    for clip_idx, (wav, targets, cls_name, subclass) in enumerate(load_nn_dataset(
        eval_cfg["url"], "test", sr, hop, n_fft,
        max_rows=max_rows, yield_subclass=True, name=eval_cfg["name"],
    )):
        wav_gpu = wav.to(device)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            probs = model(wav_gpu)  # (1, 3, T) — includes mel preprocessing
        classify_ns += time.perf_counter_ns() - t0
        T = probs.shape[-1]
        T_actual = min(T, targets.shape[-1])  # may be less than T due to center-padding in STFT

        # Per-frame label mapping: 0=speech→-1, 1=music→1, 2=background→2
        label_map = torch.tensor([-1, 1, 2], device=device)

        # Per-frame y_pred: argmax across (speech, music, background) channels
        class_probs = probs[0, :, :T_actual]          # (3, T)
        pred_idx = class_probs.argmax(dim=0)           # 0=speech, 1=music, 2=background
        y_pred_frames = label_map[pred_idx].cpu().numpy()

        # Per-frame y_true: derived from timestamp-based targets tensor.
        tgt = targets[:, :T_actual]                    # (3, T_actual)
        true_idx = tgt.argmax(dim=0)
        y_true_frames = label_map[true_idx].cpu().numpy()

        y_true_list.extend(y_true_frames.tolist())
        y_pred_list.extend(y_pred_frames.tolist())
        scores_list.append(class_probs.cpu().numpy().T)   # (T_actual, 3)
        subclasses_list.extend([subclass] * T_actual)
        clip_ids_list.append(np.full(T_actual, clip_idx, dtype=np.int64))
        total_frames += T_actual

    time_per_frame_ns = classify_ns / total_frames if total_frames else 0.0
    device_str = _device_label(device.type == "cuda")

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    y_sub = np.array(subclasses_list, dtype="U40")
    y_scores = np.vstack(scores_list) if scores_list else np.empty((0, 3))
    clip_ids = np.concatenate(clip_ids_list) if clip_ids_list else np.empty((0,), dtype=np.int64)

    return y_true, y_pred, y_sub, time_per_frame_ns, device_str, y_scores, clip_ids
