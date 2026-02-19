# svm.py
# Marek Hric

import os
import warnings
from collections import deque

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelclass import ModelClass


class SVM(ModelClass):
    name: str = "unnamed"

    def __init__(
        self,
        name: str,
    ):
        self.name = name

        self.svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=0.0001,
                        max_iter=1000,
                        tol=1e-3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        # cca 300ms / 15ms hop
        self.dec_buf = deque(maxlen=20)

    def _get_weights_path(self) -> str:
        path = f"{os.path.dirname(__file__)}/../weights/{self.name}"
        return path

    def fit(self, X: np.ndarray, y: np.ndarray):
        weights_path = self._get_weights_path()
        if os.path.exists(weights_path):
            print(f"Loading SVM weights from {weights_path}")
            self.svm = joblib.load(weights_path)
            return

        X = X.astype(np.float64)
        self.svm.fit(X, y)

    def predict(self, frame: np.ndarray) -> int:
        x = np.atleast_2d(frame.astype(np.float64))

        decision = self.svm.decision_function(x)
        self.dec_buf.append(decision[0])

        smoothed = np.mean(self.dec_buf)
        return -1 if smoothed < 0 else 1

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        return self.svm.predict(X)

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        warnings.warn("proba on SVM won't work")
        return np.array([])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        warnings.warn("proba on SVM won't work")
        return np.array([])

    def save(self):
        path = self._get_weights_path()
        print(f"Saving SVM weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "svm_pipeline": self.svm,
            "name": self.name,
        }
        joblib.dump(state, path)

    def load(self):
        path = self._get_weights_path()
        print(f"Loading SVM weights from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"SVM weights not found at {path}")

        state = joblib.load(path)
        self.svm = state["svm_pipeline"]
        self.name = state.get("name", "unnamed")
