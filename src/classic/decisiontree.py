# decisiontree.py
# Marek Hric

import logging
from collections import deque

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from .modelclass import ModelClass

log = logging.getLogger(__name__)


class DecisionTree(ModelClass):
    name: str = "unnamed"

    def __init__(self, name: str, n_last_decisions: int = 30, decision_forget_factor: float= 0.9):
        if n_last_decisions <= 0:
            raise ValueError("n_last_decisions must be positive")
        if not (0.0 < decision_forget_factor <= 1.0):
            raise ValueError("decision_forget_factor must be in (0.0, 1.0]")
        self.tree = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "selector",
                    SelectKBest(
                        score_func=f_classif, k=10
                    )
                ),
                ("classifier", DecisionTreeClassifier()),
            ]
        )
        self.name: str = name
        self.last_decisions: deque[np.ndarray] = deque(maxlen=n_last_decisions)
        self.decision_forget_factor: float = decision_forget_factor
        _indices = np.arange(n_last_decisions)
        _weights = np.exp(-_indices / self.decision_forget_factor)
        self.smoothing_weights = _weights / np.sum(_weights)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        log.info("Training DecisionTree on all 3 classes")
        self.tree.fit(X, y)

    def _state_to_save(self):
        return {
            "tree": self.tree,
            "n_last_decisions": self.last_decisions.maxlen,
            "decision_forget_factor": self.decision_forget_factor,
        }

    def _restore_state(self, state) -> None:
        self.tree = state["tree"]
        # last_decisions deque is rebuilt per-frame; smoothing_weights recomputed
        n = state.get("n_last_decisions", 30)
        f = state.get("decision_forget_factor", 0.9)
        self.last_decisions = deque(maxlen=n)
        _indices = np.arange(n)
        _weights = np.exp(-_indices / f)
        self.smoothing_weights = _weights / np.sum(_weights)

    def predict(self, frame: np.ndarray) -> int:
        probs = self.tree.predict_proba(frame.reshape(1, -1))[0]   # shape (n_classes,)
        self.last_decisions.append(probs)

        decisions = np.array(self.last_decisions)[::-1]             # (n, n_classes)
        curr_weights = self.smoothing_weights[:len(decisions)]
        curr_weights = curr_weights / curr_weights.sum()
        smoothed = curr_weights @ decisions                         # (n_classes,)

        classes = self.tree.named_steps['classifier'].classes_
        return int(classes[np.argmax(smoothed)])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        probs = self.tree.predict_proba(X)      # (N, n_classes)
        classes = self.tree.named_steps['classifier'].classes_
        N, K = probs.shape
        norm = np.convolve(np.ones(N), self.smoothing_weights, mode='full')[:N]
        smoothed = np.stack([
            np.convolve(probs[:, k], self.smoothing_weights, mode='full')[:N] / norm
            for k in range(K)
        ], axis=1)
        return classes[np.argmax(smoothed, axis=1)]

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        return self.tree.predict_proba(frame.reshape(1, -1))[0]

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict_proba(X)
