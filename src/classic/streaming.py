# classic/streaming.py
# Real-time streaming classifier for GMM / SVM / DT.

import logging
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

from .. import config
from . import MODELS, FeatExtractor

_LABEL_STR = {-1: "speech", 1: "music", 2: "inactive"}

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


def run_mic(model_name: str, device: Optional[int] = None, duration: float = 0.0) -> None:
    """Open default mic, push samples into StreamingClassifier, log predictions.

    Requires ``sounddevice``. ``duration=0`` runs until Ctrl+C.
    """
    try:
        import sounddevice as sd  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice not installed — `uv add sounddevice` or install portaudio"
        ) from exc

    classifier = StreamingClassifier(model_name)
    sr = classifier.sr
    blocksize = classifier.fh  # 15ms at 8kHz = 120 samples

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            log.warning("mic status: %s", status)
        audio_q.put(indata[:, 0].copy())

    log.info(
        "Opening mic: sr=%d, blocksize=%d (%dms hop), model=%s",
        sr, blocksize, int(1000 * blocksize / sr), model_name,
    )

    stop_flag = threading.Event()
    t_start = time.perf_counter()

    with sd.InputStream(
        samplerate=sr, channels=1, blocksize=blocksize, dtype="float32",
        device=device, callback=callback,
    ):
        log.info("Listening — Ctrl+C to stop.")
        try:
            while not stop_flag.is_set():
                if duration > 0 and (time.perf_counter() - t_start) >= duration:
                    break
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                preds = classifier.push(chunk)
                for label, proba in preds:
                    name = _LABEL_STR.get(label, str(label))
                    sys.stdout.write(f"\r{name:<8}  p={proba:.2f}")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            log.info("Stopped by user.")
    sys.stdout.write("\n")
