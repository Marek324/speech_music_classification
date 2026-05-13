# src/classic/svm.py
# Marek Hric

import logging
from collections import deque

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..seed import RAND_SEED
from .modelclass import ModelClass

log = logging.getLogger(__name__)

_MAX_PER_CLASS = 50_000


def _subsample_balanced(X: np.ndarray, y: np.ndarray, max_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Cap each class to ``max_per_class`` rows via uniform random subsampling."""
    rng = np.random.default_rng(RAND_SEED)
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
    """3-class RBF SVM with a rolling decision-vector buffer for inference smoothing."""

    name: str = "unnamed"

    def __init__(self, name: str):
        self.name = name

        self.svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1, gamma=3, cache_size=2000, random_state=RAND_SEED)),
            ]
        )
        self.dec_buf = deque(maxlen=20)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = _subsample_balanced(X, y, max_per_class=_MAX_PER_CLASS)
        self.svm.fit(X.astype(np.float64), y)

    def _state_to_save(self):
        return {"svm_pipeline": self.svm}

    def _restore_state(self, state) -> None:
        self.svm = state["svm_pipeline"]

    def predict(self, frame: np.ndarray) -> int:
        """Predict label for a single frame; smooths the OvO decision vector over the recent buffer."""
        x = np.atleast_2d(frame.astype(np.float64))

        decision = self.svm.decision_function(x)[0]   # shape (3,) for 3-class OvO
        self.dec_buf.append(decision)

        smoothed = np.mean(self.dec_buf, axis=0)       # shape (3,)
        classes = self.svm.named_steps['svm'].classes_
        return int(classes[np.argmax(smoothed)])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        return self.svm.predict(X)

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        """Softmax over the OvO decision vector for one frame; columns [speech, music, background]."""
        x = np.atleast_2d(frame.astype(np.float64))
        d = self.svm.decision_function(x)              # (1, 3) OvR-shaped
        exp_d = np.exp(d - d.max(axis=1, keepdims=True))
        return (exp_d / exp_d.sum(axis=1, keepdims=True))[0]

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Softmax over OvO decision vectors for all frames; returns (N, 3) probabilities."""
        X = X.astype(np.float64)
        d = self.svm.decision_function(X)              # (N, 3) OvR-shaped
        exp_d = np.exp(d - d.max(axis=1, keepdims=True))
        return exp_d / exp_d.sum(axis=1, keepdims=True)
