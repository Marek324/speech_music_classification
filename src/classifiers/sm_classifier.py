# sm_classifier.py
# Marek Hric

from abc import ABC, abstractmethod
import numpy as np
from sm_lib import SMDataset

class SMClassifier(ABC):
    @abstractmethod
    def fit(self, X: SMDataset):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> int:
        pass

