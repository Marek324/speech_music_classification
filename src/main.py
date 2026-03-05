# main.py
# Marek Hric

import logging
from pathlib import Path

import click
import numpy as np

from . import config
from .decisiontree import DecisionTree
from .evaluator import Evaluator
from .feat_extractor import FeatExtractor
from .gmm import GMM
from .input_handler import InputHandler, SUBCLASS_DTYPE
from .logging_config import setup_logging
from .modelclass import ModelClass
from .svm import SVM

log = logging.getLogger(__name__)


MODELS = {"decision_tree": DecisionTree, "gmm": GMM, "svm": SVM}

DEFAULT_DATASET = "Marek324/speech-music-classification"


def train(model: ModelClass):
    cfg = config.get_config()
    ds_link = cfg.get("dataset", {}).get("train", DEFAULT_DATASET)
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link=ds_link,
        ds_split="train",
    )

    X = ih.getX()
    y = ih.getY()

    model.fit(X, y)
    model.save()


def eval_model(model: ModelClass):
    model.load()

    cfg = config.get_config()
    ds_link = cfg.get("dataset", {}).get("eval", DEFAULT_DATASET)
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


def mic(model: ModelClass):
    fe = FeatExtractor()
    ih = InputHandler(mode="microphone", feat_extractor=fe)
    # while True
    # wait for frame
    # model predict


def smoke_test():
    np.random.seed(42)
    cfg = config.get_config()
    model_name = cfg["model"]["name"]

    fe = FeatExtractor()
    frame_len = fe.fl
    n_frames = 200

    # Generate synthetic audio frames and extract features
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


def dataset_stats():
    class DummyFeatExtractor(FeatExtractor):
        def extract(self, frame):
            return np.zeros(1, dtype=np.float32)

    cfg = config.get_config()
    ds_link = cfg.get("dataset", {}).get("stats", DEFAULT_DATASET)
    ih = InputHandler(
        mode="dataset",
        feat_extractor=DummyFeatExtractor(),
        ds_link=ds_link,
        ds_split="train",
        ds_revision="refs/convert/parquet",
    )

    ih.summary()


@click.group()
@click.argument("model", type=click.Choice(["decision_tree", "gmm", "svm"]))
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, model: str, verbose: bool):
    """Speech/music classifier. Usage: smclassifier <model> <command>"""
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    config.init_config(None, model_name=model)
    ctx.ensure_object(dict)
    ctx.obj["model"] = model


@cli.command("train")
@click.pass_context
def train_cmd(ctx):
    """Train the model."""
    m = MODELS[ctx.obj["model"]](ctx.obj["model"])
    train(m)


@cli.command("eval")
@click.pass_context
def eval_cmd(ctx):
    """Evaluate the model."""
    m = MODELS[ctx.obj["model"]](ctx.obj["model"])
    eval_model(m)


@cli.command("mic")
@click.pass_context
def mic_cmd(ctx):
    """Microphone mode (placeholder)."""
    m = MODELS[ctx.obj["model"]](ctx.obj["model"])
    mic(m)


@cli.command("smoke-test")
def smoke_test_cmd():
    """Run smoke test with synthetic data (no dataset download)."""
    smoke_test()


@cli.command("dataset-stats")
def dataset_stats_cmd():
    """Print dataset statistics."""
    dataset_stats()


if __name__ == "__main__":
    cli()
