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


_SCALE_CLIP = 10.0


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
            reg_covar=1e-3,
            max_iter=200,
            init_params="kmeans",
            random_state=0,
            verbose=2,
        )
        self.gmm_music = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            reg_covar=1e-3,
            max_iter=200,
            init_params="kmeans",
            random_state=0,
            verbose=2,
        )
        self.gmm_inactive = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            reg_covar=1e-3,
            max_iter=200,
            init_params="kmeans",
            random_state=0,
            verbose=2,
        )

        self.ll_buf = deque(maxlen=66)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.scaler.transform(X), -_SCALE_CLIP, _SCALE_CLIP)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        X = X.astype(np.float64)

        self.scaler.fit(X)
        Xn = self._scale(X)

        X_s = Xn[y == -1]
        X_m = Xn[y == 1]
        X_i = Xn[y == 2]

        if len(X_s) == 0 or len(X_m) == 0 or len(X_i) == 0:
            raise ValueError("Speech, music, and inactive samples are all required")

        self.gmm_speech.fit(X_s)
        self.gmm_music.fit(X_m)
        self.gmm_inactive.fit(X_i)

    def _state_to_save(self):
        return {
            "speech": self.gmm_speech,
            "music": self.gmm_music,
            "inactive": self.gmm_inactive,
            "scaler": self.scaler,
        }

    def _restore_state(self, state) -> None:
        self.gmm_speech = state["speech"]
        self.gmm_music = state["music"]
        self.gmm_inactive = state["inactive"]
        self.scaler = state["scaler"]

    def predict(self, frame: np.ndarray) -> int:
        x = frame.astype(np.float64)
        x = np.atleast_2d(x)
        x = self._scale(x)

        ll_s = self.gmm_speech.score_samples(x)[0]
        ll_m = self.gmm_music.score_samples(x)[0]
        ll_n = self.gmm_inactive.score_samples(x)[0]
        self.ll_buf.append(np.array([ll_s, ll_m, ll_n]))

        smoothed = np.mean(self.ll_buf, axis=0)
        return [-1, 1, 2][int(np.argmax(smoothed))]

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        x = frame.astype(np.float64)
        x = np.atleast_2d(x)
        x = self._scale(x)

        ll_s = self.gmm_speech.score_samples(x)
        ll_m = self.gmm_music.score_samples(x)
        ll_n = self.gmm_inactive.score_samples(x)

        ll = np.vstack([ll_s, ll_m, ll_n]).T
        ll_norm = logsumexp(ll, axis=1, keepdims=True)

        probs = np.exp(ll - ll_norm)

        # columns: [speech, music, inactive]
        return probs[0]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        x = X.astype(np.float64)
        Xn = self._scale(x)

        ll = np.vstack([
            self.gmm_speech.score_samples(Xn),
            self.gmm_music.score_samples(Xn),
            self.gmm_inactive.score_samples(Xn),
        ]).T   # (N, 3)
        return np.array([-1, 1, 2])[np.argmax(ll, axis=1)]

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        x = X.astype(np.float64)
        Xn = self._scale(x)

        ll = np.vstack([
            self.gmm_speech.score_samples(Xn),
            self.gmm_music.score_samples(Xn),
            self.gmm_inactive.score_samples(Xn),
        ]).T   # (N, 3)
        ll_norm = logsumexp(ll, axis=1, keepdims=True)

        return np.exp(ll - ll_norm)
