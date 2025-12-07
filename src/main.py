# main.py
# Marek Hric

import argparse

import config
from decisiontree import DecisionTree
from feat_extractor import FeatExtractor
from frame_dataset import FrameDataset
from input_handler import InputHandler
from modelclass import ModelClass
from evaluator import Evaluator


def main():
    cfg = config.get_config()
    fe = FeatExtractor()

    # TODO: dectree decides on selection at the end
    # if cfg.fsel is not None:
    #     fe = FeatSelector(fe)

    match cfg.model.name:
        case "decision_tree":
            input = FrameDataset(input)
            print(input.summary())
            model = DecisionTree()

    assert isinstance(model, ModelClass)

    match cfg.mode:  # will need to rethink validation in InputHandler
        case "train":
            input = InputHandler(
                "dataset",
                fe,
                ds_link="Marek324/speech-music-classification",
                ds_split="train",
            )
        case "eval":
            input: FrameDataset = FrameDataset(
                InputHandler(
                    "dataset",
                    fe,
                    ds_link="Marek324/speech-music-classification",
                    ds_split="test",
                )
            )
            evaluator = Evaluator(input)
            evaluator.eval(model)
        case "mic":
            input = InputHandler("microphone", fe)

    # input either iterabledataset or framedataset based on model

    # model.fit


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
