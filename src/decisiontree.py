# decisiontree.py
# Marek Hric

from typing import Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from frame_dataset import FrameDataset
from input_handler import InputHandler
from modelclass import ModelClass


class DecisionTree(ModelClass):
    def __init__(self):
        self.tree = DecisionTreeClassifier()

    def fit(self, X: Union[InputHandler, FrameDataset]):
        if isinstance(X, InputHandler):
            raise ValueError("DecisionTree needs FrameDataset")

        assert isinstance(X, FrameDataset)

        self.tree.fit(X.frames, X.get_all_labels())

    def predict(self, X: np.ndarray) -> int:
        return 2
