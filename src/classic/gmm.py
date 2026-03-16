# gmm.py
# Marek Hric

import logging
from collections import deque

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from .modelclass import ModelClass

log = logging.getLogger(__name__)


class GMM(ModelClass):
    name: str = "unnamed"

    def __init__(
        self,
        name: str,
    ):
        self.name = name

        self.scaler = RobustScaler()

        self.gmm_speech = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            reg_covar=1e-4,
            max_iter=200,
            init_params="kmeans",
            random_state=0,
            verbose=2,
        )
        self.gmm_music = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            reg_covar=1e-4,
            max_iter=200,
            init_params="kmeans",
            random_state=0,
            verbose=2,
        )

        self.delta_buf = deque(maxlen=66)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        X = X.astype(np.float64)

        Xn = self.scaler.fit_transform(X)

        X_s = Xn[y == -1]
        X_m = Xn[y == 1]

        if len(X_s) == 0 or len(X_m) == 0:
            raise ValueError("Both speech and music samples are required")

        self.gmm_speech.fit(X_s)
        self.gmm_music.fit(X_m)

    def _state_to_save(self):
        return {"speech": self.gmm_speech, "music": self.gmm_music, "scaler": self.scaler}

    def _restore_state(self, state) -> None:
        self.gmm_speech = state["speech"]
        self.gmm_music = state["music"]
        self.scaler = state["scaler"]

    def predict(self, frame: np.ndarray) -> int:
        x = frame.astype(np.float64)
        x = np.atleast_2d(x)
        x = self.scaler.transform(x)

        ll_s = self.gmm_speech.score_samples(x)[0]
        ll_m = self.gmm_music.score_samples(x)[0]

        delta = ll_s - ll_m
        self.delta_buf.append(delta)

        smoothed = np.mean(self.delta_buf)
        return -1 if smoothed > 0 else 1

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        x = frame.astype(np.float64)
        x = np.atleast_2d(x)
        x = self.scaler.transform(x)

        ll_s = self.gmm_speech.score_samples(x)
        ll_m = self.gmm_music.score_samples(x)

        ll = np.vstack([ll_s, ll_m]).T
        ll_norm = logsumexp(ll, axis=1, keepdims=True)

        probs = np.exp(ll - ll_norm)

        # columns: [speech, music]
        return probs[0]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        x = X.astype(np.float64)
        Xn = self.scaler.transform(x)

        ll_s = self.gmm_speech.score_samples(Xn)
        ll_m = self.gmm_music.score_samples(Xn)

        return np.where(ll_s > ll_m, -1, 1)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        x = X.astype(np.float64)
        Xn = self.scaler.transform(x)

        ll_s = self.gmm_speech.score_samples(Xn)
        ll_m = self.gmm_music.score_samples(Xn)

        ll = np.vstack([ll_s, ll_m]).T
        ll_norm = logsumexp(ll, axis=1, keepdims=True)

        return np.exp(ll - ll_norm)
