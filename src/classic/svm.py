# svm.py
# Marek Hric

import logging
import warnings
from collections import deque

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .modelclass import ModelClass

log = logging.getLogger(__name__)

_MAX_PER_CLASS = 50_000


def _subsample_balanced(X: np.ndarray, y: np.ndarray, max_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    indices = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        indices.append(idx)
    idx_all = np.concatenate(indices)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


class SVM(ModelClass):
    name: str = "unnamed"

    def __init__(self, name: str):
        self.name = name

        self.svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1, gamma=3, cache_size=2000)),
            ]
        )
        self.dec_buf = deque(maxlen=20)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        mask = np.isin(y, [-1, 1])
        X, y = X[mask], y[mask]
        X, y = _subsample_balanced(X, y, max_per_class=_MAX_PER_CLASS)
        self.svm.fit(X.astype(np.float64), y)

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
