# tcn/dataset.py
# Dataset loading and iteration for TCN train/eval.

import numpy as np
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from .config import get_config

LABEL_MAP = {"speech": 0, "music": 1, "noise": -1, "inactive": -1}  # -1 = skip


def _class_to_target(cls: str) -> torch.Tensor | None:
    """Convert class string to (2,) target tensor [speech, music]. Returns None for skip."""
    idx = LABEL_MAP.get(cls, -1)
    if idx < 0:
        return None
    t = torch.zeros(2, dtype=torch.float32)
    t[idx] = 1.0
    return t


def get_tcn_dataset(ds_link: str, split: str):
    """Load HF dataset once. Resample to TCN sample_rate."""
    cfg = get_config()
    return load_dataset(ds_link, split=split).cast_column(
        "audio",
        Audio(sampling_rate=cfg["sample_rate"], num_channels=1),
    )


def iter_tcn_rows(ds, max_rows: int | None, desc: str, yield_subclass: bool = False):
    """Iterate over HF dataset, yield (waveform, targets) or (waveform, targets, subclass) per row."""
    cfg = get_config()
    hop = cfg["hop_length"]
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
        cls_name = row["class"]
        target = _class_to_target(cls_name)
        if target is None:
            continue

        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, samples)
        n_frames = (wav.shape[-1] - cfg["n_fft"]) // hop + 1
        if n_frames < 1:
            continue
        targets = target.unsqueeze(1).expand(2, n_frames)  # (2, T)
        if yield_subclass:
            subclass = row.get("subclass", cls_name)
            yield wav, targets, subclass
        else:
            yield wav, targets


def load_tcn_dataset(
    ds_link: str, split: str, max_rows: int | None = None, yield_subclass: bool = False
):
    """Load HF dataset, yield (waveform, targets) or (waveform, targets, subclass) per row.
    max_rows=None loads full dataset."""
    ds = get_tcn_dataset(ds_link, split)
    yield from iter_tcn_rows(ds, max_rows, f"Loading {split}", yield_subclass=yield_subclass)
