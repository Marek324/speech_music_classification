# src/common.py
# Marek Hric

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

# Unified label mapping used across input_handler, evaluator, frame_label
LABEL_MAP = {"speech": -1, "music": 1, "background": 2, "inactive": 2, "noise": 2}


def subclass_primary(name: str) -> int:
    """Map a subclass name to its primary class label (-1 speech, 1 music, 2 background).

    `speech_*` → -1, `music_*` → 1, `noise`/`background*`/`inactive*` (legacy) → 2.
    """
    if name.startswith("speech"):
        return LABEL_MAP["speech"]
    if name.startswith("music"):
        return LABEL_MAP["music"]
    if name == "noise" or name.startswith("noise_") or name.startswith("background") or name.startswith("inactive"):
        return LABEL_MAP["background"]
    raise ValueError(f"Unknown subclass: {name!r}")


class EndOfDatasetException(Exception):
    def __init__(self):
        super()


@dataclass
class FrameMetadata:
    label: int  # -1, 1, 2
    rec_class: str
    subclass: str


@dataclass
class FrameData:
    audio: np.ndarray
    feats: np.ndarray
    metadata: Optional[FrameMetadata]


@dataclass
class FrameMetadataStrLabel:
    label: str  # -1, 1, 2
    rec_class: str
    subclass: str


@dataclass
class FrameDataStrLabel:
    audio: np.ndarray
    feats: np.ndarray
    metadata: Optional[FrameMetadataStrLabel]


def frame_label(
    anns: List[Dict[str, Union[str, int]]], f_start: int, f_end: int
) -> int:
    """Return integer label for a frame: majority class by overlap with annotations."""
    overlap = {"speech": 0, "music": 0, "background": 0}

    # Accept both "background" (canonical) and "inactive" (legacy) annotation keys.
    label_aliases = [("speech", "speech"), ("music", "music"), ("background", "background"), ("inactive", "background")]
    for ann in anns:
        for ann_key, canon in label_aliases:
            ran = ann.get(ann_key)
            if ran is not None:
                start = max(f_start, ran["start"])
                end = min(f_end, ran["end"])
                if start < end:
                    overlap[canon] += end - start

    if sum(overlap.values()) == 0:
        return LABEL_MAP["background"]

    majority = max(overlap.items(), key=lambda x: x[1])[0]
    return LABEL_MAP[majority]


def frame_label_str(
    anns: List[Dict[str, Union[str, int]]], f_start: int, f_end: int
) -> str:
    """Return string label for a frame: majority class by overlap with annotations."""
    overlap = {"speech": 0, "music": 0, "background": 0}

    for ann in anns:
        lbl = ann.get("label")
        # Normalize legacy "inactive" label to canonical "background".
        if lbl == "inactive":
            lbl = "background"
        if lbl not in ["speech", "music", "background"]:
            raise ValueError(f"Unknown label: {lbl}")

        start = max(f_start, ann["start"])
        end = min(f_end, ann["end"])
        if start < end:
            overlap[lbl] += end - start

    if sum(overlap.values()) == 0:
        return "background"

    majority = max(overlap.items(), key=lambda x: x[1])[0]
    return majority


def ms_to_samples(ms: int, sr: int) -> int:
    return int(ms * sr / 1000)


def get_num_workers() -> int:
    """Return number of worker processes: all available CPUs, capped at 64."""
    return min(os.cpu_count() or 1, 64)
