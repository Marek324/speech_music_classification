# classic/cli.py
# CLI commands for classic hand-crafted feature models.

import logging
import time

import click
import numpy as np

from .. import config
from ..evaluator import run_evaluation
from ..input_handler import InputHandler, SUBCLASS_DTYPE
from . import MODELS, FeatExtractor
from .evaluation import eval_classic

log = logging.getLogger(__name__)

CLASSIC_MODELS = ("decision_tree", "gmm", "svm")


def _train_classic(model_name: str):
    config.init_config(None, model_name=model_name)
    cfg = config.get_config()
    train_cfg = cfg["dataset"]["train"]
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link=train_cfg["url"],
        ds_split="train",
        ds_revision=train_cfg["revision"],
    )

    X = ih.getX()
    y = ih.getY()

    model = MODELS[model_name](model_name)
    model.fit(X, y)
    model.save()


def _eval_classic(model_name: str):
    eval_classic(model_name, save_to_file=True)


def _mic_classic(model_name: str):
    config.init_config(None, model_name=model_name)
    fe = FeatExtractor()
    _ih = InputHandler(mode="microphone", feat_extractor=fe)
    # placeholder - real-time mic not implemented


def _smoke_test_classic(model_name: str):
    config.init_config(None, model_name=model_name)
    np.random.seed(42)
    fe = FeatExtractor()
    frame_len = fe.fl
    n_frames = 200

    frames = np.random.randn(n_frames, frame_len).astype(np.float32) * 0.1
    X_list = []
    for i in range(n_frames):
        feat = fe.extract(frames[i])
        X_list.append(feat)
    X = np.array(X_list, dtype=np.float64)

    y = np.random.choice([-1, 1], n_frames)
    subclasses = np.array(["smoke"] * n_frames, dtype=SUBCLASS_DTYPE)

    model = MODELS[model_name](f"smoke_{model_name}")
    model.fit(X, y)

    t0 = time.perf_counter_ns()
    y_pred = model.predict_batch(X)
    time_per_sample_ns = (time.perf_counter_ns() - t0) / len(X)

    res = run_evaluation(y, y_pred, subclasses, time_per_sample_ns, output_name=model.name, save_to_file=False)
    log.info(
        "Smoke test passed. Model: %s, FeatExtractor OK | 2-class F1: %.4f | 3-class F1: %.4f",
        model_name,
        res.two_class.f1,
        res.three_class.f1,
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
        ds_link=cfg["dataset"]["stats"]["url"],
        ds_split="train",
        ds_revision=cfg["dataset"]["stats"]["revision"],
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
    def mic_cmd():
        _mic_classic(model_name)


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
