# gmm.py
# Marek Hric


import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from modelclass import ModelClass


class GMM(ModelClass):
    name: str = "unnamed"

    def __init__(self, name: str):
        self.gmm = GaussianMixture()
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray):
        print("Training Decisiongmm")
        self.gmm.fit(X, y)

    def predict(self, frame: np.ndarray) -> int:
        return self.gmm.predict(frame)

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        return np.array([])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.gmm.predict(X)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([])

    def save(self):
        joblib.dump(self.gmm, f"{self.name}")

    def load(self):
        self.gmm = joblib.load(f"{self.name}")
