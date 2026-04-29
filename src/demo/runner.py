"""Uniform streaming wrapper over classic + NN speech/music classifiers.

Each runner exposes ``sr``, ``chunk_samples``, and ``push(samples)`` returning
a list of ``Prediction(t, label)`` where ``label`` is the display label:
−1 speech, 0 inactive, +1 music. The raw classic label space
(−1 / 1 / 2) is remapped once here so the UI never sees label ``2``.

Each runner keeps its own frame counter and emits absolute timestamps from
the start of the stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .weights_check import CLASSIC_MODELS, NN_MODELS


@dataclass
class Prediction:
    t: float
    label: int  # -1 speech, 0 inactive, +1 music


# Raw classic labels {-1, 1, 2} → display labels {-1, 1, 0}
_CLASSIC_REMAP = {-1: -1, 1: 1, 2: 0}

# NN channel order in model output: (speech, music, inactive)
_NN_CHANNEL_LABELS = (-1, 1, 0)


class Runner(Protocol):
    sr: int
    chunk_samples: int

    def push(self, samples: np.ndarray) -> list[Prediction]: ...
    def close(self) -> None: ...


class ClassicRunner:
    """Wraps classic StreamingClassifier.

    Resets the global config singleton in __init__ so different classic models
    (each with its own buffer settings — e.g. gmm/svm use sr=8000 but
    decision_tree uses sr=16000) can be used interchangeably in one process.
    """

    def __init__(self, model_name: str) -> None:
        if model_name not in CLASSIC_MODELS:
            raise ValueError(f"unknown classic model: {model_name}")

        from .. import config as _cfg_mod
        _cfg_mod._cfg = None

        from ..classic.streaming import StreamingClassifier
        self._cls = StreamingClassifier(model_name)
        self.sr: int = int(self._cls.sr)
        self.chunk_samples: int = int(self._cls.fh)
        self._frame_period = self._cls.fh / self._cls.sr
        self._frames = 0

    def push(self, samples: np.ndarray) -> list[Prediction]:
        raw = self._cls.push(samples)
        out: list[Prediction] = []
        for label, _proba in raw:
            out.append(
                Prediction(
                    t=self._frames * self._frame_period,
                    label=_CLASSIC_REMAP[label],
                )
            )
            self._frames += 1
        return out

    def close(self) -> None:
        # Drop the StreamingClassifier so its sklearn pipeline (scaler +
        # estimator) becomes unreachable; the user wants RAM back on Stop.
        self._cls = None


class NNRunner:
    """Wraps StreamingInference around a trained TCN / TCN-LSTM / SmallTCN."""

    def __init__(self, variant_name: str) -> None:
        if variant_name not in NN_MODELS:
            raise ValueError(f"unknown NN variant: {variant_name}")

        import torch
        from safetensors.torch import load_file

        if variant_name == "tcn":
            from ..nn.tcn.config import (
                get_config,
                get_preprocess_stats_path,
                get_weights_path,
            )
            from ..nn.tcn.model import SpeechMusicDetector
            from ..nn.tcn.streaming import StreamingInference

            cfg = get_config()
            weights_path = get_weights_path()
            stats_path = get_preprocess_stats_path()
            model = SpeechMusicDetector(cfg=cfg, stats_path=stats_path)
        else:
            from ..nn.tcn.streaming import StreamingInference
            from ..nn.variants import VARIANTS

            v = VARIANTS[variant_name]
            cfg = v.get_config()
            weights_path = v.get_weights_path()
            stats_path = v.get_stats_path()
            model = v.cls(cfg=cfg, stats_path=stats_path)

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path} — run "
                "`uv run python scripts/download_artifacts.py` first."
            )

        state = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state)
        model.eval()
        self._model = model
        self._torch = torch
        self._inference = StreamingInference(
            model, hop_length=cfg["hop_length"], sample_rate=cfg["sample_rate"]
        )

        self.sr: int = int(cfg["sample_rate"])
        # One inference per hop keeps per-frame timestamps accurate and bounds
        # compute cost (streaming inference re-runs the TCN over its receptive
        # field each call — bigger chunks would be wasteful).
        self.chunk_samples: int = int(cfg["hop_length"])
        self._frame_period = self.chunk_samples / self.sr
        self._frames = 0

    def push(self, samples: np.ndarray) -> list[Prediction]:
        out: list[Prediction] = []
        idx = 0
        n = samples.shape[0]
        while idx + self.chunk_samples <= n:
            chunk = samples[idx : idx + self.chunk_samples]
            tensor = self._torch.from_numpy(np.ascontiguousarray(chunk)).float()
            p_speech, p_music, p_inactive = self._inference.process_chunk(tensor)
            probs = (p_speech, p_music, p_inactive)
            label = _NN_CHANNEL_LABELS[int(np.argmax(probs))]
            out.append(
                Prediction(
                    t=self._frames * self._frame_period,
                    label=label,
                )
            )
            self._frames += 1
            idx += self.chunk_samples
        return out

    def close(self) -> None:
        # Break refs to the torch model + StreamingInference (which holds the
        # streaming buffer and a back-reference to the model). Without this the
        # weights stay resident after Stop and switching models doubles RAM.
        self._inference = None
        self._model = None
        self._torch = None


def _variant_runners() -> dict[str, Callable[[], Runner]]:
    # Imported lazily so module import stays cheap; the demo loads torch via
    # NNRunner construction, not via this dict's existence.
    from ..nn.variants import VARIANTS
    # Each lambda binds `name` via default-arg trick to avoid late-binding bugs.
    return {name: (lambda n=name: NNRunner(n)) for name in VARIANTS}


RUNNERS: dict[str, Callable[[], Runner]] = {
    "decision_tree": lambda: ClassicRunner("decision_tree"),
    "gmm": lambda: ClassicRunner("gmm"),
    "svm": lambda: ClassicRunner("svm"),
    "tcn": lambda: NNRunner("tcn"),
    **_variant_runners(),
}
