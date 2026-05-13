# nn/dataset.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

import io
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

LABEL_MAP = {"speech": 0, "music": 1, "noise": 2, "background": 2, "inactive": 2}


def _class_to_target(cls: str) -> torch.Tensor:
    """Convert class string to (3,) target tensor [speech, music, background]."""
    idx = LABEL_MAP.get(cls, 2)
    t = torch.zeros(3, dtype=torch.float32)
    t[idx] = 1.0
    return t


def _timestamps_to_frame_labels(labels_list, n_frames, sample_rate, hop_length) -> torch.Tensor:
    """Convert timestamp annotations to per-frame labels.

    Returns (n_frames,) int tensor: 0=speech, 1=music, -1=background.
    Uncovered frames default to -1 (background).
    """
    frame_labels = torch.full((n_frames,), -1, dtype=torch.long)
    label_name_to_idx = {"speech": 0, "music": 1}
    for entry in labels_list:
        label = entry.get("label", "")
        if label not in label_name_to_idx:
            continue
        idx = label_name_to_idx[label]
        start_frame = int(entry["start"] / 1000 * sample_rate / hop_length)
        end_frame = int(entry["end"] / 1000 * sample_rate / hop_length)
        start_frame = max(0, min(start_frame, n_frames))
        end_frame = max(0, min(end_frame, n_frames))
        frame_labels[start_frame:end_frame] = idx
    return frame_labels


def _decode_audio_bytes(raw: dict, target_sr: int) -> np.ndarray:
    """Decode a local-parquet `{"bytes","path"}` audio struct → mono float32 @ target_sr."""
    import soundfile as sf

    if raw.get("bytes") is not None:
        data, src_sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32", always_2d=False)
    elif raw.get("path"):
        data, src_sr = sf.read(raw["path"], dtype="float32", always_2d=False)
    else:
        raise ValueError("audio struct has neither 'bytes' nor 'path'")
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    if src_sr != target_sr:
        import librosa
        data = librosa.resample(
            data.astype(np.float32, copy=False),
            orig_sr=src_sr, target_sr=target_sr, res_type="polyphase",
        )
    return np.ascontiguousarray(data, dtype=np.float32)


def _is_local_dataset(ds_link: str) -> bool:
    """True iff ``ds_link`` resolves to an existing local directory of parquet shards."""
    try:
        return Path(ds_link).expanduser().is_dir()
    except OSError:
        return False


def _local_parquet_shards(ds_root: Path, split: str) -> list[str]:
    """Return sorted parquet shard paths for a split, raising if none exist."""
    shards = sorted(ds_root.joinpath(split).glob("*/part_*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No parquet shards found under {ds_root / split}/*/part_*.parquet"
        )
    return [str(s) for s in shards]


def get_nn_dataset(ds_link: str, split: str, sample_rate: int, name: str = "full"):
    """Load dataset once. Resample to ``sample_rate``.

    Two routes:
    - HF hub identifier (e.g. ``Marek324/speech-music-classification``) — load
      via ``load_dataset(ds_link, name=name, split=split)`` and cast the audio
      column to ``Audio(sampling_rate=sample_rate)`` so HF resamples on access.
    - Local directory laid out as ``{ds_link}/{split}/{modality}/part_*.parquet``
      (the staging output of ``scripts/dataset/build.py``) — load the parquet
      shards directly, leave the audio column as a raw ``{bytes,path}`` struct,
      and let ``iter_nn_rows`` decode + resample bytes via soundfile + librosa.
      The local route deliberately avoids HF's Audio decode path because that
      pulls in torchcodec, which fails to load on hosts with FFmpeg 8 / no
      CUDA NPP libs.
    """
    if _is_local_dataset(ds_link):
        shards = _local_parquet_shards(Path(ds_link).expanduser(), split)
        ds = load_dataset("parquet", data_files={split: shards}, split=split)
        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate, decode=False))
        return ds
    return load_dataset(ds_link, name=name, split=split).cast_column(
        "audio",
        Audio(sampling_rate=sample_rate, decode=False),
    )


def iter_nn_rows(
    ds,
    max_rows: int | None,
    desc: str,
    sr: int,
    hop: int,
    n_fft: int,
    yield_subclass: bool = False,
):
    """Iterate over HF dataset, yield (waveform, targets) or (waveform, targets, subclass) per row.

    Uses per-frame timestamp labels when available (row["labels"]), otherwise falls back to
    clip-level row["class"]. Noise/background clips yield [0,0] targets instead of being skipped.
    """
    for i, row in enumerate(tqdm(ds, desc=desc)):
        if max_rows and i >= max_rows:
            break
        raw = row["audio"]
        if isinstance(raw, dict):
            if "array" in raw:
                audio = raw["array"]
            else:
                audio = _decode_audio_bytes(raw, target_sr=sr)
        elif hasattr(raw, "get_all_samples"):
            audio = raw.get_all_samples().data
            if hasattr(audio, "cpu"):
                audio = audio.cpu()
            audio = np.asarray(audio).squeeze()
        else:
            audio = np.asarray(raw).squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        n_frames = (wav.shape[-1] - n_fft) // hop + 1
        if n_frames < 1:
            continue

        labels_raw = row.get("labels") or []
        if isinstance(labels_raw, dict):
            keys = list(labels_raw.keys())
            labels_list = [
                {k: labels_raw[k][i] for k in keys}
                for i in range(len(labels_raw[keys[0]]))
            ]
        else:
            labels_list = labels_raw
        if labels_list:
            frame_labels = _timestamps_to_frame_labels(labels_list, n_frames, sr, hop)
            targets = torch.zeros(3, n_frames, dtype=torch.float32)
            targets[0, frame_labels == 0] = 1.0
            targets[1, frame_labels == 1] = 1.0
            targets[2, frame_labels == -1] = 1.0
        else:
            cls_name = row["class"]
            target = _class_to_target(cls_name)
            targets = target.unsqueeze(1).expand(-1, n_frames)

        cls_name = row["class"]
        if yield_subclass:
            subclass = row.get("subclass", cls_name)
            yield wav, targets, cls_name, subclass
        else:
            yield wav, targets


def load_nn_dataset(
    ds_link: str,
    split: str,
    sr: int,
    hop: int,
    n_fft: int,
    max_rows: int | None = None,
    yield_subclass: bool = False,
    name: str = "full",
):
    """Load HF dataset, yield (waveform, targets) or (waveform, targets, subclass) per row.
    max_rows=None loads full dataset."""
    ds = get_nn_dataset(ds_link, split, sr, name=name)
    yield from iter_nn_rows(ds, max_rows, f"Loading {split}", sr, hop, n_fft, yield_subclass=yield_subclass)
