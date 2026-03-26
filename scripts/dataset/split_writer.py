"""Parquet shards: ``{config_dir}/{split}/{modality}/part_*.parquet`` (LibriSpeech-style tier + HF splits)."""

import io
from pathlib import Path
from typing import Any, Union

import numpy as np
import soundfile as sf
from datasets import Audio
from pydub import AudioSegment
from torchcodec.decoders import AudioDecoder

from labeling import SR

ROWS_PER_PARQUET_FILE = 1024


def _audio_to_wav_bytes(
    audio: Union[AudioDecoder, dict, np.ndarray],
) -> bytes:
    buf = io.BytesIO()
    if isinstance(audio, np.ndarray):
        sf.write(buf, audio, samplerate=SR, format="WAV")
    elif isinstance(audio, AudioDecoder):
        data = audio.get_all_samples().data
        if hasattr(data, "cpu"):
            data = data.cpu()
        sf.write(buf, data.numpy().squeeze(), samplerate=SR, format="WAV")
    else:
        seg = (
            AudioSegment.from_file(io.BytesIO(audio["bytes"]))
            if audio["bytes"] is not None
            else AudioSegment.from_file(audio["path"])
        )
        seg = seg.set_frame_rate(SR).set_channels(1)
        seg.export(buf, format="wav")
    return buf.getvalue()


class SplitWriter:
    """Writes under one Hub **config** folder (e.g. ``.../full/``): ``{split}/{modality}/``."""

    def __init__(self, modality: str, split: str, config_dir: Path):
        self.modality = modality
        self.split = split
        self.base = config_dir / split / modality
        self.base.mkdir(parents=True, exist_ok=True)
        self._shard_idx = len(list(self.base.glob("part_*.parquet")))
        self._rows: dict[str, list[Any]] = {
            "audio": [],
            "sampling_rate": [],
            "class": [],
            "subclass": [],
            "source": [],
            "row_idx": [],
            "labels": [],
        }

    def _flush(self) -> None:
        if not self._rows["audio"]:
            return
        import datasets as hf
        features = hf.Features({
            "audio": Audio(sampling_rate=SR),
            "sampling_rate": hf.Value("int32"),
            "class": hf.Value("string"),
            "subclass": hf.Value("string"),
            "source": hf.Value("string"),
            "row_idx": hf.Value("int64"),
            "labels": hf.Sequence(hf.Features({
                "label": hf.Value("string"),
                "start": hf.Value("int32"),
                "end": hf.Value("int32"),
            })),
        })
        ds = hf.Dataset.from_dict(self._rows, features=features)
        path = self.base / f"part_{self._shard_idx:05d}.parquet"
        ds.to_parquet(str(path))
        self._shard_idx += 1
        for k in self._rows:
            self._rows[k].clear()

    def close(self) -> None:
        self._flush()

    def write(
        self,
        audio: Union[AudioDecoder, dict, np.ndarray],
        labels: list[dict],
        name: str,
        idx: int,
        cls: str,
        subclass: str,
    ) -> None:
        wav_bytes = _audio_to_wav_bytes(audio)
        self._rows["audio"].append({"bytes": wav_bytes, "path": None})
        self._rows["sampling_rate"].append(SR)
        self._rows["class"].append(cls)
        self._rows["subclass"].append(subclass)
        self._rows["source"].append(name)
        self._rows["row_idx"].append(idx)
        # HF Sequence(Features) expects columnar dict-of-lists, not list-of-dicts
        self._rows["labels"].append({
            "label": [l["label"] for l in labels],
            "start": [l["start"] for l in labels],
            "end": [l["end"] for l in labels],
        })
        if len(self._rows["audio"]) >= ROWS_PER_PARQUET_FILE:
            self._flush()
