# classic/evaluation.py
# Evaluation interface for classic models. Handles data loading and inference.

import logging
import time

import numpy as np

from ..evaluator import run_evaluation, _device_label
from ..input_handler import InputHandler

from . import MODELS, FeatExtractor

log = logging.getLogger(__name__)


def get_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load model and test data, run inference. Returns (y_true, y_pred, subclasses, time_per_sample_ns).
    Labels: -1=speech, 1=music, 2=inactive.
    """
    from .. import config

    config.init_config(None, model_name=model_name)
    model = MODELS[model_name](model_name)
    model.load()

    cfg = config.get_config()
    eval_cfg = cfg["dataset"]["eval"]
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link=eval_cfg["url"],
        ds_split="test",
        ds_name=eval_cfg.get("name"),
    )

    X = ih.getX()
    y = ih.getY()
    subclasses = ih.getSubclasses()

    # Sequential extraction benchmark: time fe.extract() on a dummy frame
    # (avoids parallel-timing underestimate from dataset.map(num_proc=N))
    fe.reset()
    dummy_frame = np.zeros(fe.fl, dtype=np.float32)
    _bench = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        fe.extract(dummy_frame)
        _bench.append(time.perf_counter_ns() - t0)
    extract_time_ns = float(np.median(_bench))

    t0 = time.perf_counter_ns()
    y_pred = model.predict_batch(X)
    classify_ns = time.perf_counter_ns() - t0
    n = len(X)
    time_per_sample_ns = (classify_ns / n) + extract_time_ns

    return y, y_pred, subclasses, time_per_sample_ns


def eval_classic(model_name: str, save_to_file: bool = True):
    """Evaluate classic model on test split. Uses main evaluator for metrics and output."""
    y_true, y_pred, subclasses, time_per_sample_ns = get_predictions(model_name)
    return run_evaluation(
        y_true,
        y_pred,
        subclasses,
        time_per_sample_ns,
        output_name=model_name,
        save_to_file=save_to_file,
        device=_device_label(False),
    )
