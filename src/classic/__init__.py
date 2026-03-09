# classic/__init__.py
# Marek Hric
# run by uv run classic <model> <command>

from .decisiontree import DecisionTree
from .feat_extractor import FeatExtractor
from .gmm import GMM
from .svm import SVM

MODELS = {"decision_tree": DecisionTree, "gmm": GMM, "svm": SVM}

__all__ = ["FeatExtractor", "MODELS"]
