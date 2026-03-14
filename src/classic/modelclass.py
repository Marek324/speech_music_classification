# modelclass.py
# Marek Hric

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np

log = logging.getLogger(__name__)


class ModelClass(ABC):
    """Base class for speech/music classifiers."""

    name: str

    def _get_weights_path(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "weights" / self.name

    def _weights_exist(self) -> bool:
        return self._get_weights_path().exists()

    def _prepare_save_path(self) -> Path:
        path = self._get_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            log.info("Weights already exist at %s, overwriting", path)
        return path

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        self._train(X, y)

    @abstractmethod
    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model from scratch."""
        pass

    def save(self) -> None:
        path = self._prepare_save_path()
        log.info("Saving %s weights to %s", self.__class__.__name__, path)
        joblib.dump(self._state_to_save(), path)

    def load(self) -> None:
        path = self._get_weights_path()
        if not path.exists():
            raise FileNotFoundError(f"Weights not found at {path}")
        log.info("Loading %s weights from %s", self.__class__.__name__, path)
        self._restore_state(joblib.load(path))

    @abstractmethod
    def _state_to_save(self) -> Any:
        """Return object to persist (passed to joblib.dump)."""
        pass

    @abstractmethod
    def _restore_state(self, state: Any) -> None:
        """Restore model from loaded state."""
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> int:
        pass

    @abstractmethod
    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        pass
