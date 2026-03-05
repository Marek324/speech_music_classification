# svm.py
# Marek Hric

import logging
import warnings
from collections import deque

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modelclass import ModelClass

log = logging.getLogger(__name__)


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

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        X = X.astype(np.float64)
        self.svm.fit(X, y)

    def _state_to_save(self):
        return {"svm_pipeline": self.svm}

    def _restore_state(self, state) -> None:
        self.svm = state["svm_pipeline"]

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
        warnings.warn("SVM does not provide probabilities; returning placeholder")
        return np.array([0.5, 0.5])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        warnings.warn("SVM does not provide probabilities; returning placeholder")
        return np.full((len(X), 2), 0.5)
