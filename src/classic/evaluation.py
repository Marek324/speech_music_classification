# classic/evaluation.py
# Evaluation interface for classic models. Handles data loading and inference.

import logging
import time

import numpy as np

from ..evaluator import run_evaluation, _device_label
from ..input_handler import InputHandler

from . import MODELS, FeatExtractor

log = logging.getLogger(__name__)


def get_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Load model and test data, run inference.
    Returns (y_true, y_pred, subclasses, time_per_sample_ns, y_scores).
    Labels: -1=speech, 1=music, 2=inactive.
    y_scores: (N, 3) — columns [speech, music, inactive].
    """
    from .. import config

    config.init_config(None, model_name=model_name)
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
    y_scores = model.predict_proba_batch(X)  # (N, 3) — [speech, music, inactive]
    n = len(X)
    time_per_sample_ns = (classify_ns / n) + extract_time_ns

    return y, y_pred, subclasses, time_per_sample_ns, y_scores


def eval_classic(model_name: str, save_to_file: bool = True):
    """Evaluate classic model on test split. Uses main evaluator for metrics and output."""
    y_true, y_pred, subclasses, time_per_sample_ns, y_scores = get_predictions(model_name)
    return run_evaluation(
        y_true,
        y_pred,
        subclasses,
        time_per_sample_ns,
        output_name=model_name,
        save_to_file=save_to_file,
        device=_device_label(False),
        y_scores=y_scores,
    )
