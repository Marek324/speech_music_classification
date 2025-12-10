# gmm.py
# Marek Hric


import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from modelclass import ModelClass


class GMM(ModelClass):
    name: str = "unnamed"

    def __init__(self, name: str):
        self.tree = GaussianMixture()
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray):
        print("Training DecisionTree")
        self.tree.fit(X, y)

    def predict(self, frame: np.ndarray) -> int:
        return self.tree.predict(frame)

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        return np.array([])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([])

    def save(self):
        joblib.dump(self.tree, f"{self.name}")

    def load(self):
        self.tree = joblib.load(f"{self.name}")
