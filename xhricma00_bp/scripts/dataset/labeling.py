# scripts/dataset/labeling.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""Timestamp labels for speech / music / background (Silero VAD, pyannote + RMS music, full-clip background)."""

import io
import os
from typing import Any, Optional, Union

import numpy as np
import torch

torch.backends.nnpack.set_flags(False)

from pydub import AudioSegment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from silero_vad import get_speech_timestamps, load_silero_vad

try:
    from torchcodec.decoders import AudioDecoder
except (ImportError, RuntimeError):
    class AudioDecoder:
        """Sentinel — torchcodec unavailable; no real instances exist."""

SR = 16_000

PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"

_MUSIC_RMS_HOP_MS = 20
_MUSIC_RMS_PERCENTILE = 20
_MUSIC_RMS_FLOOR_SCALE = 0.35


def make_label(label: str, start: int, end: int) -> dict[str, Any]:
    return {"label": label, "start": start, "end": end}


def _ms_from_samples(n_samples: int) -> int:
    return int(n_samples * 1000 / SR)


def _pydub_from_hf_dict(audio: dict) -> AudioSegment:
    raw = (
        AudioSegment.from_file(io.BytesIO(audio["bytes"]))
        if audio["bytes"] is not None
        else AudioSegment.from_file(audio["path"])
    )
    return raw.set_frame_rate(SR).set_channels(1)


def _wav_mono_16_pydub(seg: AudioSegment) -> np.ndarray:
    samples = np.asarray(seg.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (8 * seg.sample_width - 1))
    return samples


def _wav_mono_16k_from_hf_dict(audio: dict) -> np.ndarray:
    return _wav_mono_16_pydub(_pydub_from_hf_dict(audio))


def _duration_ms(audio: Union[AudioDecoder, dict, np.ndarray]) -> int:
    if isinstance(audio, np.ndarray):
        return int(len(audio) * 1000 / SR)
    if isinstance(audio, AudioDecoder):
        return int(audio.get_all_samples().duration_seconds * 1000)
    return len(_pydub_from_hf_dict(audio))


def _silero_wav_and_length(audio: Union[AudioDecoder, np.ndarray]) -> tuple[Optional[torch.Tensor], int]:
    if isinstance(audio, AudioDecoder):
        try:
            samples = audio.get_all_samples()
        except RuntimeError:
            return None, 0
        wav = samples.data
        return wav, int(wav.shape[-1])
    arr = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    return arr, int(arr.shape[-1])


def _labels_from_silero_ts(timestamps: list[dict], n_samples: int) -> list[dict[str, Any]]:
    if not timestamps:
        return [make_label("background", 0, _ms_from_samples(n_samples))]

    labels: list[dict[str, Any]] = []
    prev_end = 0
    for ts in timestamps:
        s, e = int(ts["start"]), int(ts["end"])
        if s > prev_end:
            labels.append(make_label("background", _ms_from_samples(prev_end), _ms_from_samples(s)))
        labels.append(make_label("speech", _ms_from_samples(s), _ms_from_samples(e)))
        prev_end = e
    if prev_end < n_samples:
        labels.append(make_label("background", _ms_from_samples(prev_end), _ms_from_samples(n_samples)))
    return labels


def _per_hop_rms(wav: np.ndarray, hop: int) -> np.ndarray:
    n_hop = (wav.shape[0] + hop - 1) // hop
    out = np.empty(n_hop, dtype=np.float64)
    for i in range(n_hop):
        sl = wav[i * hop : (i + 1) * hop]
        out[i] = float(np.sqrt(np.mean(sl * sl) + 1e-12))
    return out


def _rle_hop_bool_to_music_labels(mask: np.ndarray, hop: int, n_samples: int) -> list[dict[str, Any]]:
    """True -> music, False -> background."""
    if mask.size == 0:
        return [make_label("background", 0, _ms_from_samples(n_samples))]

    labels: list[dict[str, Any]] = []
    cur = bool(mask[0])
    start_i = 0
    for j in range(1, mask.size):
        v = bool(mask[j])
        if v != cur:
            t0, t1 = start_i * hop, min(j * hop, n_samples)
            labels.append(make_label("music" if cur else "background", _ms_from_samples(t0), _ms_from_samples(t1)))
            start_i, cur = j, v
    labels.append(
        make_label(
            "music" if cur else "background",
            _ms_from_samples(start_i * hop),
            _ms_from_samples(n_samples),
        )
    )
    return labels


def _fill_speech_hop_mask(speech_ann: Any, hop: int, n_hop: int) -> np.ndarray:
    speech_hop = np.zeros(n_hop, dtype=bool)
    for segment, _track, label in speech_ann.itertracks(yield_label=True):
        if str(label) != "SPEECH":
            continue
        a = int(segment.start * SR / hop)
        b = int(np.ceil(segment.end * SR / hop))
        a, b = max(0, min(a, n_hop)), max(0, min(b, n_hop))
        if b > a:
            speech_hop[a:b] = True
    return speech_hop


class VADLabeler:
    def __init__(self) -> None:
        self.model = load_silero_vad()

    def label(self, audio: Union[AudioDecoder, np.ndarray]) -> Optional[list[dict[str, Any]]]:
        wav, n = _silero_wav_and_length(audio)
        if wav is None:
            return None
        return _labels_from_silero_ts(get_speech_timestamps(wav, self.model), n)


class MusicLabeler:
    """Music vs background: pyannote SPEECH ∪ RMS loud frames (instrumental)."""

    def __init__(self) -> None:
        model = Model.from_pretrained(PYANNOTE_SEGMENTATION_MODEL)
        pl = VoiceActivityDetection(segmentation=model)
        pl.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
        self._pipeline: VoiceActivityDetection = pl

    def label(self, audio: Union[dict, np.ndarray]) -> Optional[list[dict[str, Any]]]:
        """Return music/background timestamp labels via pyannote SPEECH ∪ RMS gating."""
        try:
            if isinstance(audio, np.ndarray):
                wav = audio
            else:
                wav = _wav_mono_16k_from_hf_dict(audio)
        except Exception:
            return None

        n = wav.shape[0]
        if n == 0:
            return [make_label("background", 0, 0)]

        hop = max(1, int(SR * _MUSIC_RMS_HOP_MS / 1000))
        n_hop = (n + hop - 1) // hop
        rms = _per_hop_rms(wav, hop)
        noise_floor = float(np.percentile(rms, _MUSIC_RMS_PERCENTILE) * _MUSIC_RMS_FLOOR_SCALE)
        loud = rms > max(noise_floor, 1e-8)

        py_file: dict[str, Any] = {
            "uri": "music_chunk",
            "waveform": torch.from_numpy(wav).unsqueeze(0),
            "sample_rate": SR,
        }
        try:
            speech_ann = self._pipeline(py_file)
        except Exception:
            return None

        music_hop = _fill_speech_hop_mask(speech_ann, hop, n_hop) | loud
        return _rle_hop_bool_to_music_labels(music_hop, hop, n)


class SilenceLabeler:
    def label(self, audio: Union[AudioDecoder, dict, np.ndarray]) -> list[dict[str, Any]]:
        return [make_label("background", 0, _duration_ms(audio))]


_LABELER_CACHE: dict[str, Union[VADLabeler, MusicLabeler, SilenceLabeler]] = {}

_LABELER_FACTORY: dict[str, type[VADLabeler] | type[MusicLabeler] | type[SilenceLabeler]] = {
    "vad": VADLabeler,
    "music": MusicLabeler,
    "silence": SilenceLabeler,
}


def get_labeler(detector: str) -> Union[VADLabeler, MusicLabeler, SilenceLabeler]:
    """Return a cached labeler instance for ``detector`` ∈ {vad, music, silence}."""
    if detector not in _LABELER_CACHE:
        factory = _LABELER_FACTORY.get(detector)
        if factory is None:
            raise ValueError(f"Unknown detector: {detector!r}")
        _LABELER_CACHE[detector] = factory()
    return _LABELER_CACHE[detector]
