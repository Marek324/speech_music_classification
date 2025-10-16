# sm_classifier.py
# Marek Hric

from abc import ABC, abstractmethod

class SMClassifier(ABC):
    @abstractmethod
    def fit(self, X: SMDataset):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> int:
        pass

