# classic/cli.py
# CLI commands for classic hand-crafted feature models.

import logging
import time

import click
import numpy as np

from .. import config
from ..evaluator import run_evaluation
from ..input_handler import InputHandler, SUBCLASS_DTYPE
from ..seed import seed_all
from . import MODELS, FeatExtractor
from .evaluation import eval_classic

log = logging.getLogger(__name__)

CLASSIC_MODELS = ("decision_tree", "gmm", "svm")


def _train_classic(model_name: str):
    seed_all()
    config.init_config(None, model_name=model_name)
    cfg = config.get_config()
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link=cfg["dataset"]["url"],
        ds_split="train",
        ds_name=cfg["dataset"].get("name"),
    )

    X = ih.getX()
    y = ih.getY()

    model = MODELS[model_name](model_name)
    model.fit(X, y)
    model.save()


def _eval_classic(model_name: str):
    eval_classic(model_name, save_to_file=True)


def _mic_classic(model_name: str, device: int | None, duration: float):
    from .streaming import run_mic
    run_mic(model_name, device=device, duration=duration)


def _smoke_test_classic(model_name: str):
    config.init_config(None, model_name=model_name)
    seed_all()
    fe = FeatExtractor()

    if model_name in ("gmm", "svm"):
        n_segments = 60
        segments = np.random.randn(n_segments, fe.sr).astype(np.float32) * 0.1
        X_list = [fe.extract_segment(segments[i]) for i in range(n_segments)]
        X = np.array(X_list, dtype=np.float64)
        n_samples = n_segments
    else:
        n_frames = 200
        frames = np.random.randn(n_frames, fe.fl).astype(np.float32) * 0.1
        X_list = []
        for i in range(n_frames):
            feat = fe.extract(frames[i])
            X_list.append(feat)
        X = np.array(X_list, dtype=np.float64)
        n_samples = n_frames

    y = np.random.choice([-1, 1, 2], n_samples)
    subclasses = np.array(["smoke"] * n_samples, dtype=SUBCLASS_DTYPE)

    model = MODELS[model_name](f"smoke_{model_name}")
    model.fit(X, y)

    t0 = time.perf_counter_ns()
    y_pred = model.predict_batch(X)
    time_per_frame_ns = (time.perf_counter_ns() - t0) / len(X)

    res = run_evaluation(y, y_pred, subclasses, time_per_frame_ns, output_name=model.name, save_to_file=False)
    log.info(
        "Smoke test passed. Model: %s, FeatExtractor OK | 3-class F1: %.4f",
        model_name,
        res.f1,
    )


def _dataset_stats_classic():
    config.init_config(None, model_name="decision_tree")
    class DummyFeatExtractor(FeatExtractor):
        def extract(self, frame):
            return np.zeros(1, dtype=np.float32)

    cfg = config.get_config()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=DummyFeatExtractor(),
        ds_link=cfg["dataset"]["url"],
        ds_split="train",
        ds_name=cfg["dataset"].get("name"),
    )

    ih.summary()


def _add_model_commands(group: click.Group, model_name: str):
    """Add train, eval, smoke-test to a model group."""

    @group.command("train")
    def train():
        _train_classic(model_name)

    @group.command("eval")
    def eval_cmd():
        _eval_classic(model_name)

    @group.command("smoke-test")
    def smoke_test():
        _smoke_test_classic(model_name)

    @group.command("mic")
    @click.option("--device", type=int, default=None, help="Input device index (sounddevice).")
    @click.option("--duration", type=float, default=0.0, help="Seconds to run; 0 = until Ctrl+C.")
    def mic_cmd(device: int | None, duration: float):
        _mic_classic(model_name, device=device, duration=duration)


@click.group("classic")
def classic_group():
    """Classic hand-crafted feature models (DT, GMM, SVM)."""
    pass


for _model in CLASSIC_MODELS:
    _model_grp = click.Group(_model)
    _add_model_commands(_model_grp, _model)
    classic_group.add_command(_model_grp, _model)


@classic_group.command("dataset-stats")
def dataset_stats_cmd():
    """Print dataset statistics (uses decision_tree config for buffers)."""
    _dataset_stats_classic()
