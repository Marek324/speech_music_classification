# modelclass.py
# Marek Hric

from abc import ABC, abstractmethod

import numpy as np


class ModelClass(ABC):
    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> int:
        pass

    @abstractmethod
    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
