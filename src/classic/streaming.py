# src/classic/streaming.py
# Marek Hric
# Streaming classifier for GMM / SVM / DT — consumed by src/demo/.

import logging

import numpy as np

from .. import config
from . import MODELS, FeatExtractor

log = logging.getLogger(__name__)


class StreamingClassifier:
    """Push raw audio samples; get a predicted label per frame hop.

    Wraps a ``FeatExtractor`` in its frame-by-frame mode (``extract()``)
    and a loaded classic model, so one call to ``push`` consumes any
    number of new samples and yields one prediction per completed hop.
    """

    def __init__(self, model_name: str) -> None:
        config.init_config(None, model_name=model_name)
        self.fe = FeatExtractor()
        self.model = MODELS[model_name](model_name)
        self.model.load()
        self.sr = self.fe.sr
        self.fl = self.fe.fl
        self.fh = self.fe.fh
        self._pending = np.zeros(0, dtype=np.float32)
        self._bootstrapped = False

    def reset(self) -> None:
        self.fe.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._bootstrapped = False

    def push(self, samples: np.ndarray) -> list[tuple[int, float]]:
        """Consume new samples and return ``[(label, proba_max), ...]``.

        One entry per newly-completed 15ms hop.
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        self._pending = np.concatenate([self._pending, samples])
        out: list[tuple[int, float]] = []

        # First frame needs fl samples; subsequent frames hop by fh.
        if not self._bootstrapped:
            if self._pending.size < self.fl:
                return out
            frame = self._pending[: self.fl]
            self._pending = self._pending[self.fh :]  # keep overlap for next hop
            feats = self.fe.extract(frame)
            label = int(self.model.predict(feats))
            proba = self._max_proba(feats)
            out.append((label, proba))
            self._bootstrapped = True

        while self._pending.size >= self.fl:
            frame = self._pending[: self.fl]
            self._pending = self._pending[self.fh :]
            feats = self.fe.extract(frame)
            label = int(self.model.predict(feats))
            proba = self._max_proba(feats)
            out.append((label, proba))
        return out

    def _max_proba(self, feats: np.ndarray) -> float:
        try:
            p = self.model.predict_proba(feats)
            return float(np.max(np.asarray(p)))
        except Exception:
            return float("nan")
