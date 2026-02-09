# main.py
# Marek Hric

import argparse
from pathlib import Path

import numpy as np

import config
from decisiontree import DecisionTree
from evaluator import Evaluator
from feat_extractor import FeatExtractor
from gmm import GMM
from input_handler import InputHandler
from modelclass import ModelClass
from svm import SVM

DATASET = "Marek324/speech-music-classification-tmp"
TRAIN_DATASET = DATASET
EVAL_DATASET = DATASET
STATS_DATASET = "Marek324/speech-music-classification-test"
# STATS_DATASET=DATASET


def train(model):
    assert isinstance(model, ModelClass)
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link=TRAIN_DATASET,
        ds_split="train",
    )

    X = ih.getX()
    y = ih.getY()

    model.fit(X, y)
    model.save()


def eval(model: ModelClass):
    model.load()

    fe = FeatExtractor()
    ih = InputHandler(
        "dataset",
        fe,
        ds_link=EVAL_DATASET,
        ds_split="test",
    )

    evaluator = Evaluator()
    res = evaluator.eval(model, ih.getX(), ih.getY(), ih.getSubclasses(), n_classes=2)
    print(res)
    res = evaluator.eval(model, ih.getX(), ih.getY(), ih.getSubclasses(), n_classes=3)
    print(res)


def mic(model):
    fe = FeatExtractor()
    ih = InputHandler(mode="microphone", feat_extractor=fe)
    # while True
    # wait for frame
    # model predict


def dataset_stats():
    class DummyFeatExtractor(FeatExtractor):
        def extract(self, frame):
            # Returns a valid numpy array so downstream code doesn't break
            return np.zeros(1, dtype=np.float32)

    ih = InputHandler(
        mode="dataset",
        feat_extractor=DummyFeatExtractor(),
        ds_link=STATS_DATASET,
        ds_split="train",
        ds_revision="refs/convert/parquet",
    )

    ih.summary()


def main(mode: str):
    cfg = config.get_config()
    assert cfg is not None

    match cfg["model"]["name"]:
        case "decision_tree":
            model = DecisionTree("decision_tree")
        case "gmm":
            model = GMM("gmm")
        case "svm":
            model = SVM("svm")
        case _:
            raise ValueError("Model doesn't exist")
    assert isinstance(model, ModelClass)

    match mode:
        case "train":
            train(model)
        case "eval":
            eval(model)
        case "mic":
            mic(model)
        case "dataset_stats":
            dataset_stats()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "mode",
        choices=["train", "eval", "run", "dataset_stats"],
        help="Mode of execution: train, eval, run",
    )

    argparser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        help="Path to toml config file",
    )

    args = argparser.parse_args()
    config.init_config(args.config_file)

    main(args.mode)
