# input_handler.py
# Marek Hric

import hashlib
import json
import logging
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm

from . import config
from .common import (
    FrameDataStrLabel,
    FrameMetadataStrLabel,
    LABEL_MAP,
    frame_label_str,
)
from .classic.feat_extractor import FeatExtractor

log = logging.getLogger(__name__)

SUBCLASS_DTYPE = "U40"


def _pad_frame(frame: np.ndarray, target_len: int, dtype=np.float32) -> np.ndarray:
    """Pad or truncate frame to target length."""
    out = np.zeros(target_len, dtype=dtype)
    out[: len(frame)] = frame[:target_len]
    return out


def _safe_feats(feats: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with zeros."""
    if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
        return np.zeros_like(feats)
    return feats


class InputHandler:
    """
    InputHandler operates in two modes:
        - "dataset"     → load HF dataset, extract features, aggregate to X, y
        - "microphone"  → real-time mic stream (placeholder, not implemented)
    """

    def __init__(
        self,
        mode: str,
        feat_extractor: FeatExtractor,
        ds_link: str = "",
        ds_split: str = "train",
        ds_revision: str = "main",
        ds_name: Optional[str] = None,
    ):
        if mode not in ("dataset", "microphone"):
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.fextractor = feat_extractor

        cfg = config.get_config()
        self.sr = cfg["sample_rate"]
        self.channels = cfg["channels"]
        buf = cfg["buffers"]
        self.frame_len = buf["frame_length_ms"] * self.sr // 1000
        self.hop_len = buf["hop_length_ms"] * self.sr // 1000

        self.st_buffer: Optional[np.ndarray] = None
        self.extract_duration_ns: int = 0

        if mode == "dataset":
            self._init_dataset_mode(ds_link, ds_split, ds_revision, ds_name)
        else:
            self._init_microphone_mode()

    def _cache_key(self, ds_link: str, ds_split: str, ds_revision: str, ds_name: Optional[str] = None) -> str:
        cfg = config.get_config()
        fingerprint = json.dumps({
            "ds_link": ds_link,
            "ds_split": ds_split,
            "ds_revision": ds_revision,
            "ds_name": ds_name,
            "model": cfg["model"]["name"],
            "sample_rate": cfg["sample_rate"],
            "n_fft": cfg["n_fft"],
            "buffers": cfg["buffers"],
            "features": cfg["features"],
        }, sort_keys=True)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _cache_path(self, ds_link: str, ds_split: str, ds_revision: str, ds_name: Optional[str] = None) -> "Path":
        from pathlib import Path
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{self._cache_key(ds_link, ds_split, ds_revision, ds_name)}.npz"

    def _init_dataset_mode(
            self, ds_link: str, ds_split: str, ds_revision: str, ds_name: Optional[str] = None
    ) -> None:
        if not ds_link:
            raise ValueError("Dataset mode requires ds_link")

        self.ds_stats: Dict[str, Any] = {
            "frames": 0,
            "classes": Counter(),
            "subclasses": Counter(),
        }

        cache_path = self._cache_path(ds_link, ds_split, ds_revision, ds_name)
        if cache_path.exists():
            log.info("Loading features from cache: %s", cache_path)
            data = np.load(cache_path, allow_pickle=False)
            self.X = data["X"]
            self.y = data["y"]
            self.subclasses = data["subclasses"].astype(SUBCLASS_DTYPE)
            self.extract_duration_ns = 0
            log.info("Loaded %d frames from cache.", len(self.X))
            return

        dataset = load_dataset(
            ds_link, name=ds_name, split=ds_split, revision=ds_revision
        ).cast_column(
            "audio",
            Audio(sampling_rate=self.sr, num_channels=self.channels),
        )
        assert isinstance(dataset, Dataset)

        t0 = time.perf_counter_ns()
        processed = dataset.map(
            self._process_row,
            desc="Extracting frames/features",
            num_proc=os.cpu_count(),
            load_from_cache_file=True,
        )
        self.extract_duration_ns = time.perf_counter_ns() - t0

        row_lengths = [len(lst) for lst in processed["labels"]]
        total_frames = sum(row_lengths)
        feat_dim = len(processed[0]["feats"][0])

        log.info("Allocating X: (%s, %s)", total_frames, feat_dim)

        self.X = np.empty((total_frames, feat_dim), dtype=np.float32)
        self.y = np.empty(total_frames, dtype=int)
        self.subclasses = np.empty(total_frames, dtype=SUBCLASS_DTYPE)

        cursor = 0
        for i, row in enumerate(tqdm(processed, desc="Filling Arrays")):
            n = row_lengths[i]
            self.X[cursor : cursor + n] = np.array(row["feats"], dtype=np.float32)
            self.y[cursor : cursor + n] = [LABEL_MAP[label] for label in row["labels"]]
            self.subclasses[cursor : cursor + n] = row["subclasses"]

            self.ds_stats["frames"] += n
            self.ds_stats["classes"].update(row["labels"])
            self.ds_stats["subclasses"].update(row["subclasses"])
            cursor += n

        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        log.info("Aggregation done.")

        log.info("Saving features to cache: %s", cache_path)
        np.savez(cache_path, X=self.X, y=self.y, subclasses=np.array(self.subclasses))

    def _init_microphone_mode(self) -> None:
        self.st_buffer = np.zeros(0, dtype=np.float32)

    def get_extract_time_per_frame_ns(self) -> float:
        """Extraction time per frame (ns), or 0 if not applicable."""
        x = getattr(self, "X", None)
        if x is None or len(x) == 0 or self.extract_duration_ns == 0:
            return 0.0
        return float(self.extract_duration_ns) / len(x)

    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        self.fextractor.reset()
        audio = row["audio"].get_all_samples().data
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        audio = np.asarray(audio).squeeze()

        cls_name = row["class"]
        subclass = row["subclass"]
        labels_raw = row["labels"] or []
        if isinstance(labels_raw, dict):
            keys = list(labels_raw.keys())
            labels = [
                {k: labels_raw[k][i] for k in keys}
                for i in range(len(labels_raw[keys[0]]))
            ]
        else:
            labels = labels_raw

        feats, labels_out, subclasses_out = [], [], []
        frame_start = 0
        n_samples = len(audio)

        while frame_start + self.frame_len <= n_samples:
            frame = _pad_frame(
                audio[frame_start : frame_start + self.frame_len],
                self.frame_len,
            )
            fd = self._process_frame(frame, frame_start, labels, cls_name, subclass)

            feats.append(_safe_feats(fd.feats))
            labels_out.append(fd.metadata.label)
            assert fd.metadata is not None
            subclasses_out.append(fd.metadata.subclass)
            frame_start += self.hop_len

        return {"feats": feats, "labels": labels_out, "subclasses": subclasses_out}

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_start: int,
        labels: List[Dict],
        cls_name: str,
        subclass: str,
    ) -> FrameDataStrLabel:
        feats = self.fextractor.extract(frame)
        label = frame_label_str(
            labels, frame_start, frame_start + self.frame_len
        )
        meta = FrameMetadataStrLabel(label=label, rec_class=cls_name, subclass=subclass)
        return FrameDataStrLabel(frame, feats, meta)

    def getX(self) -> np.ndarray:
        return self.X

    def getY(self) -> np.ndarray:
        return self.y

    def getSubclasses(self) -> np.ndarray:
        return self.subclasses

    def summary(self) -> None:
        if self.mode != "dataset":
            log.info("Input Handler summary: %s mode, nothing to sum", self.mode)
            return

        total_frames = self.ds_stats["frames"]
        total_seconds = (total_frames * self.hop_len) / self.sr
        total_minutes = total_seconds / 60

        lines = [
            "Input Handler summary:",
            f"Total Frames:   {total_frames:,}",
            f"Total Duration: {total_minutes:.2f} min  ({total_seconds:.2f} sec)",
            "-" * 60,
            f"{'CLASS':<25} | {'FRAMES':<10} | {'MINUTES':<10} | {'%':<5}",
            "-" * 60,
        ]
        for lbl, count in self.ds_stats["classes"].items():
            minutes = (count * self.hop_len) / self.sr / 60
            perc = (count / total_frames) * 100
            lines.append(f"{lbl:<25} | {count:<10,} | {minutes:<10.2f} | {perc:>5.1f}%")
        lines.extend([
            "-" * 72,
            f"{'SUBCLASS':<42} | {'FRAMES':<10} | {'MINUTES':<10}",
            "-" * 72,
        ])
        for sub, count in self.ds_stats["subclasses"].most_common():
            minutes = (count * self.hop_len) / self.sr / 60
            lines.append(f"{sub:<42} | {count:<10,} | {minutes:<10.2f}")

        log.info("\n%s", "\n".join(lines))
