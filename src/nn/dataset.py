# nn/dataset.py
# Dataset loading and iteration for NN models — shared across TCN and future models.
# Config params (sr, hop, n_fft) are passed explicitly; no model-specific config dependency.

import numpy as np
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

LABEL_MAP = {"speech": 0, "music": 1, "noise": 2, "inactive": 2}


def _class_to_target(cls: str) -> torch.Tensor:
    """Convert class string to (3,) target tensor [speech, music, inactive]."""
    idx = LABEL_MAP.get(cls, 2)  # unknown classes → inactive
    t = torch.zeros(3, dtype=torch.float32)
    t[idx] = 1.0
    return t


def _timestamps_to_frame_labels(labels_list, n_frames, sample_rate, hop_length) -> torch.Tensor:
    """Convert timestamp annotations to per-frame labels.

    Returns (n_frames,) int tensor: 0=speech, 1=music, -1=inactive.
    Uncovered frames default to -1 (inactive).
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


def get_nn_dataset(ds_link: str, split: str, sample_rate: int, name: str = "full"):
    """Load HF dataset once. Resample to given sample_rate."""
    return load_dataset(ds_link, name=name, split=split).cast_column(
        "audio",
        Audio(sampling_rate=sample_rate, num_channels=1),
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
    clip-level row["class"]. Noise/inactive clips yield [0,0] targets instead of being skipped.
    """
    for i, row in enumerate(tqdm(ds, desc=desc)):
        if max_rows and i >= max_rows:
            break
        raw = row["audio"]
        if isinstance(raw, dict):
            audio = raw["array"]
        elif hasattr(raw, "get_all_samples"):
            audio = raw.get_all_samples().data
            if hasattr(audio, "cpu"):
                audio = audio.cpu()
            audio = np.asarray(audio).squeeze()
        else:
            audio = np.asarray(raw).squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, samples)
        n_frames = (wav.shape[-1] - n_fft) // hop + 1
        if n_frames < 1:
            continue

        labels_raw = row.get("labels") or []
        # Normalize dict-of-lists → list-of-dicts
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
            targets[0, frame_labels == 0] = 1.0   # speech frames
            targets[1, frame_labels == 1] = 1.0   # music frames
            targets[2, frame_labels == -1] = 1.0  # inactive: frames not covered by any annotation
        else:
            cls_name = row["class"]
            target = _class_to_target(cls_name)
            targets = target.unsqueeze(1).expand(-1, n_frames)

        cls_name = row["class"]
        if yield_subclass:
            subclass = row.get("subclass", cls_name)
            yield wav, targets, subclass
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
