# svm.py
# Marek Hric

import os
from collections import deque

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler

from modelclass import ModelClass


class SVM(ModelClass):
    name: str = "unnamed"

    def __init__(
        self,
        name: str,
    ):
        self.name = name

        self.scaler = RobustScaler()

        self.svm = SVC(
            kernel="rbf",
            gamma=3.0,
            C=1.0,
            random_state=0,
            probability=True,
            verbose=True,
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
            self.tree = joblib.load(weights_path)
            return

        X = X.astype(np.float64)
        Xn = self.scaler.fit_transform(X)

        self.svm.fit(Xn, y)

    def predict(self, frame: np.ndarray) -> int:
        x = frame.astype(np.float64)
        x = np.atleast_2d(x)
        x = self.scaler.transform(x)[0]

        decision = self.svm.decision_function(x)

        self.dec_buf.append(decision)

        smoothed = np.mean(self.dec_buf)

        return -1 if smoothed < 0 else 1

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        X_scaled = self.scaler.transform(X)
        predictions = self.svm.predict(X_scaled)
        return predictions

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        x = frame.astype(np.float64)
        x = np.atleast_2d(x)
        x_scaled = self.scaler.transform(x)
        probs = self.svm.predict_proba(x_scaled)
        return probs[0]

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)

        X_scaled = self.scaler.transform(X)

        probs = self.svm.predict_proba(X_scaled)

        return probs

    def save(self):
        path = self._get_weights_path()
        print(f"Saving GMM weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            print(f"GMM weights already exist at {path}, overwriting.")
        state = {
            "speech": self.gmm_speech,
            "music": self.gmm_music,
            "scaler": self.scaler,
        }
        joblib.dump(state, path)

    def load(self):
        path = self._get_weights_path()
        print(f"Loading GMM weights from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"GMM weights not found at {path}")

        state = joblib.load(path)
        self.gmm_speech = state["speech"]
        self.gmm_music = state["music"]
        self.scaler = state["scaler"]
