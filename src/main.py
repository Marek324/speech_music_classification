# main.py
# Marek Hric

import logging

import click
import numpy as np

from . import config
from .classic import MODELS, FeatExtractor
from .tcn.cli import eval_tcn, smoke_test_tcn, train_tcn
from .evaluator import Evaluator
from .input_handler import InputHandler, SUBCLASS_DTYPE
from .logging_config import setup_logging

log = logging.getLogger(__name__)

CLASSIC_MODELS = ("decision_tree", "gmm", "svm")


def _train_classic(model_name: str):
    config.init_config(None, model_name=model_name)
    cfg = config.get_config()
    ds_link = cfg["dataset"]["train"]
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link=ds_link,
        ds_split="train",
    )

    X = ih.getX()
    y = ih.getY()

    model = MODELS[model_name](model_name)
    model.fit(X, y)
    model.save()


def _eval_classic(model_name: str):
    config.init_config(None, model_name=model_name)
    model = MODELS[model_name](model_name)
    model.load()

    cfg = config.get_config()
    ds_link = cfg["dataset"]["eval"]
    fe = FeatExtractor()
    ih = InputHandler(
        "dataset",
        fe,
        ds_link=ds_link,
        ds_split="test",
    )

    evaluator = Evaluator()
    res = evaluator.eval(
        model,
        ih.getX(),
        ih.getY(),
        ih.getSubclasses(),
        extract_time_per_frame_ns=ih.get_extract_time_per_frame_ns(),
    )
    log.info("\n%s", res)


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
    evaluator = Evaluator()
    res = evaluator.eval(model, X, y, subclasses, save_to_file=False)
    log.info(
        "Smoke test passed. Model: %s, FeatExtractor OK | 2-class Acc: %.4f | 3-class Acc: %.4f",
        model_name,
        res.two_class.accuracy,
        res.three_class.accuracy,
    )


def _dataset_stats_classic():
    config.init_config(None, model_name="decision_tree")
    class DummyFeatExtractor(FeatExtractor):
        def extract(self, frame):
            return np.zeros(1, dtype=np.float32)

    cfg = config.get_config()
    ds_link = cfg["dataset"]["stats"]
    ih = InputHandler(
        mode="dataset",
        feat_extractor=DummyFeatExtractor(),
        ds_link=ds_link,
        ds_split="train",
        ds_revision="refs/convert/parquet",
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


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Speech/music classifier. Usage: smclassifier <approach> <model> <command>"""
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)


@cli.group()
def classic():
    """Classic hand-crafted feature models (DT, GMM, SVM)."""
    pass


# Model groups under classic
for _model in CLASSIC_MODELS:
    model_grp = click.Group(_model)
    _add_model_commands(model_grp, _model)
    classic.add_command(model_grp, _model)


@classic.command("dataset-stats")
def dataset_stats_cmd():
    """Print dataset statistics (uses decision_tree config for buffers)."""
    _dataset_stats_classic()


@cli.group()
def tcn():
    """Causal TCN for online speech/music classification (Lemaire & Holzapfel, ISMIR 2019)."""
    pass


@tcn.command("train")
@click.option("--epochs", "-e", default=3, help="Training epochs")
@click.option("--max-rows", default=None, type=int, help="Limit train rows (for quick tests)")
def tcn_train(epochs, max_rows):
    """Train the TCN model."""
    train_tcn(epochs=epochs, max_train_rows=max_rows)


@tcn.command("eval")
@click.option("--max-rows", default=100, type=int, help="Max test rows to evaluate")
def tcn_eval(max_rows):
    """Evaluate the TCN model."""
    eval_tcn(max_rows=max_rows)


@tcn.command("smoke-test")
def tcn_smoke_test():
    """Run TCN smoke test (forward pass with synthetic audio)."""
    smoke_test_tcn()
