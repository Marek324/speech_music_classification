# modelclass.py
# Marek Hric

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from frame_dataset import FrameDataset
from input_handler import InputHandler


class ModelClass(ABC):
    @abstractmethod
    def fit(self, X: Union[InputHandler, FrameDataset]):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> int:
        pass
