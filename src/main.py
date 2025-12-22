# main.py
# Marek Hric

import argparse

import numpy as np

import config
from decisiontree import DecisionTree
from evaluator import Evaluator
from feat_extractor import FeatExtractor
from gmm import GMM
from input_handler import InputHandler
from modelclass import ModelClass


def train(model):
    assert isinstance(model, ModelClass)
    fe = FeatExtractor()
    ih = InputHandler(
        mode="dataset",
        feat_extractor=fe,
        ds_link="Marek324/speech-music-classification",
        ds_split="train",
    )

    X = ih.getX()
    y = ih.getY()

    # AGGRESSIVE DEBUGGING
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"X min: {np.min(X)}, max: {np.max(X)}")
    print(f"Contains NaN: {np.any(np.isnan(X))}")
    print(f"Contains Inf: {np.any(np.isinf(X))}")
    print(f"Contains +Inf: {np.any(np.isposinf(X))}")
    print(f"Contains -Inf: {np.any(np.isneginf(X))}")

    # Find problematic rows
    inf_rows = np.where(np.any(np.isinf(X), axis=1))[0]
    if len(inf_rows) > 0:
        print(f"\nFound {len(inf_rows)} rows with infinity!")
        print(f"First few problematic rows: {inf_rows[:10]}")
        for row_idx in inf_rows[:3]:
            inf_cols = np.where(np.isinf(X[row_idx]))[0]
            print(f"  Row {row_idx}, columns with inf: {inf_cols}")
            print(f"  Values: {X[row_idx][inf_cols]}")

    # NUCLEAR OPTION: Force clean the data
    print("\nCleaning data...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Check again
    print(f"After cleaning - Contains Inf: {np.any(np.isinf(X))}")
    print(f"After cleaning - X min: {np.min(X)}, max: {np.max(X)}")

    # Check for values too large for float32
    max_float32 = np.finfo(np.float32).max
    too_large = np.any(np.abs(X) > max_float32)
    print(f"Contains values > float32 max: {too_large}")
    if too_large:
        print(f"Max absolute value: {np.max(np.abs(X))}")
        X = np.clip(X, -max_float32, max_float32)

    model.fit(X, y)
    model.save()


def eval(model: ModelClass):
    model.load()

    fe = FeatExtractor()
    ih = InputHandler(
        "dataset",
        fe,
        ds_link="Marek324/speech-music-classification",
        ds_split="test",
    )

    evaluator = Evaluator()
    res = evaluator.eval(model, ih.getX(), ih.getY(), ih.getSubclasses())
    print(res)


def mic(model):
    fe = FeatExtractor()
    ih = InputHandler(mode="microphone", feat_extractor=fe)
    # while True
    # wait for frame
    # model predict


def main():
    cfg = config.get_config()

    # TODO: dectree decides on selection at the end
    # if cfg.fsel is not None:
    #     fe = FeatSelector(fe)

    match cfg.model.name:
        case "decision_tree":
            model = DecisionTree("decision_tree")
        case "gmm":
            model = GMM("gmm")
        case _:
            raise ValueError("Model doesn't exist")
    assert isinstance(model, ModelClass)

    match cfg.mode:  # will need to rethink validation in InputHandler
        case "train":
            train(model)
        case "eval":
            eval(model)
        case "mic":
            mic(model)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "mode",
        choices=["train", "eval", "run"],
        help="Mode of execution: train, eval, run",
    )
    argparser.add_argument(
        "model",
        nargs="?",
        default="decision_tree",
        choices=["decision_tree"],
        help="Model name to use (default: decision_tree)",
    )

    argparser.add_argument(
        "-c",
        "--config-file",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose",
    )
    config.init_config(vars(argparser.parse_args()))
    cfg = config.get_config()
    print(cfg.model)
    print(cfg.fext)
    print(cfg.fsel)

    main()

    # TODO: train saves model on disk
    # eval/run loads from disk, don't forget to check if exists
