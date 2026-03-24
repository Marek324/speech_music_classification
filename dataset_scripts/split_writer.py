"""Parquet shards: ``{config_dir}/{split}/{modality}/part_*.parquet`` (LibriSpeech-style tier + HF splits)."""

import io
import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
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
            "audio_wav": [],
            "sampling_rate": [],
            "class": [],
            "subclass": [],
            "source": [],
            "row_idx": [],
            "labels_json": [],
        }

    def _flush(self) -> None:
        if not self._rows["audio_wav"]:
            return
        table = pa.table(
            {
                "audio_wav": pa.array(self._rows["audio_wav"], type=pa.binary()),
                "sampling_rate": pa.array(self._rows["sampling_rate"], type=pa.int32()),
                "class": pa.array(self._rows["class"], type=pa.string()),
                "subclass": pa.array(self._rows["subclass"], type=pa.string()),
                "source": pa.array(self._rows["source"], type=pa.string()),
                "row_idx": pa.array(self._rows["row_idx"], type=pa.int64()),
                "labels_json": pa.array(self._rows["labels_json"], type=pa.string()),
            }
        )
        path = self.base / f"part_{self._shard_idx:05d}.parquet"
        pq.write_table(table, path)
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
        self._rows["audio_wav"].append(wav_bytes)
        self._rows["sampling_rate"].append(SR)
        self._rows["class"].append(cls)
        self._rows["subclass"].append(subclass)
        self._rows["source"].append(name)
        self._rows["row_idx"].append(idx)
        self._rows["labels_json"].append(json.dumps(labels))
        if len(self._rows["audio_wav"]) >= ROWS_PER_PARQUET_FILE:
            self._flush()
