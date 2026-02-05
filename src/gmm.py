# gmm.py
# Marek Hric

import os
from collections import deque

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from modelclass import ModelClass


class GMM(ModelClass):
    name: str = "unnamed"

    def __init__(
        self,
        name: str,
    ):
        self.name = name

        self.scaler = StandardScaler()

        self.gmm_speech = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            reg_covar=1e-8,
            max_iter=200,
            init_params="kmeans",
            random_state=0,
        )
        self.gmm_music = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            reg_covar=1e-8,
            max_iter=200,
            init_params="kmeans",
            random_state=0,
        )

        # cca 300ms / 15ms hop
        self.delta_buf = deque(maxlen=20)


    def _get_weights_path(self) -> str:
        path = f"{os.path.dirname(__file__)}/../weights/{self.name}"
        return path

    def fit(self, X: np.ndarray, y: np.ndarray):
        weights_path = self._get_weights_path()
        if os.path.exists(weights_path):
            print(f"Loading GMM weights from {weights_path}")
            self.tree = joblib.load(weights_path)
            return

        Xn = self.scaler.fit_transform(X)

        X_s = Xn[y == -1]
        X_m = Xn[y == 1]

        if len(X_s) == 0 or len(X_m) == 0:
            raise ValueError("Both speech and music samples are required")

        self.gmm_speech.fit(X_s)
        self.gmm_music.fit(X_m)

    def predict(self, frame: np.ndarray) -> int:
        x = np.atleast_2d(frame)
        x = self.scaler.transform(x)

        ll_s = self.gmm_speech.score_samples(x)[0]
        ll_m = self.gmm_music.score_samples(x)[0]

        delta = ll_s - ll_m
        self.delta_buf.append(delta)

        smoothed = np.mean(self.delta_buf)
        return -1 if smoothed > 0 else 1

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(frame)
        x = self.scaler.transform(x)

        ll_s = self.gmm_speech.score_samples(x)
        ll_m = self.gmm_music.score_samples(x)

        ll = np.vstack([ll_s, ll_m]).T
        ll_norm = logsumexp(ll, axis=1, keepdims=True)

        probs = np.exp(ll - ll_norm)

        # columns: [speech, music]
        return probs[0]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        Xn = self.scaler.transform(X)

        ll_s = self.gmm_speech.score_samples(Xn)
        ll_m = self.gmm_music.score_samples(Xn)

        return np.where(ll_s > ll_m, -1, 1)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        Xn = self.scaler.transform(X)

        ll_s = self.gmm_speech.score_samples(Xn)
        ll_m = self.gmm_music.score_samples(Xn)

        ll = np.vstack([ll_s, ll_m]).T
        ll_norm = logsumexp(ll, axis=1, keepdims=True)

        return np.exp(ll - ll_norm)

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

