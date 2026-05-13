# src/classic/evaluation.py
# Marek Hric
# Evaluation interface for classic models. Handles data loading and inference.

import logging
import time

import numpy as np

from ..evaluator import run_evaluation, _device_label
from ..input_handler import InputHandler

from . import MODELS, FeatExtractor

log = logging.getLogger(__name__)


def get_predictions(
    model_name: str,
    dataset_override: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Load model and test data, run inference.
    Returns (y_true, y_pred, subclasses, time_per_frame_ns, y_scores, clip_ids).
    Labels: -1=speech, 1=music, 2=background.
    y_scores: (N, 3) — columns [speech, music, background].
    clip_ids: (N,) int — per-frame source-clip index.

    *dataset_override*: optional ``{"url", "name"}`` dict that replaces the
    ``[dataset]`` section in the loaded config — used by ``src/exp/critical/``
    to point eval at a non-default tier.
    """
    from .. import config

    if dataset_override is not None:
        config.reset_config()
    config.init_config(None, model_name=model_name, dataset_override=dataset_override)
    model = MODELS[model_name](model_name)
    model.load()

    cfg = config.get_config()
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link=cfg["dataset"]["url"],
        ds_split="test",
        ds_name=cfg["dataset"].get("name"),
    )

    X = ih.getX()
    y = ih.getY()
    subclasses = ih.getSubclasses()
    clip_ids = ih.getClipIds()

    # Sequential extraction benchmark: time feature extraction on a dummy input
    # (avoids parallel-timing underestimate from dataset.map(num_proc=N))
    fe.reset()
    _bench = []
    if model_name in ("gmm", "svm"):
        dummy_seg = np.zeros(fe.sr, dtype=np.float32)  # 1s segment
        for _ in range(20):
            t0 = time.perf_counter_ns()
            fe.extract_segment(dummy_seg)
            _bench.append(time.perf_counter_ns() - t0)
    else:
        dummy_frame = np.zeros(fe.fl, dtype=np.float32)
        for _ in range(200):
            t0 = time.perf_counter_ns()
            fe.extract(dummy_frame)
            _bench.append(time.perf_counter_ns() - t0)
    extract_time_ns = float(np.median(_bench))

    t0 = time.perf_counter_ns()
    y_pred = model.predict_batch(X)
    classify_ns = time.perf_counter_ns() - t0
    y_scores = model.predict_proba_batch(X)  # (N, 3) — [speech, music, background]
    n = len(X)
    time_per_frame_ns = (classify_ns / n) + extract_time_ns

    return y, y_pred, subclasses, time_per_frame_ns, y_scores, clip_ids


def eval_classic(
    model_name: str,
    save_to_file: bool = True,
    dataset_override: dict | None = None,
    output_name: str | None = None,
    output_dir=None,
):
    """Evaluate classic model on test split. Uses main evaluator for metrics and output.

    *dataset_override* / *output_name* / *output_dir* are forwarded through so
    experiment harnesses (e.g. ``src/exp/critical/``) can re-route eval to a
    non-default tier and a non-default results directory.
    """
    y_true, y_pred, subclasses, time_per_frame_ns, y_scores, clip_ids = get_predictions(
        model_name, dataset_override=dataset_override,
    )
    return run_evaluation(
        y_true,
        y_pred,
        subclasses,
        time_per_frame_ns,
        output_name=output_name or model_name,
        save_to_file=save_to_file,
        device=_device_label(False),
        output_dir=output_dir,
        y_scores=y_scores,
        clip_ids=clip_ids,
    )
