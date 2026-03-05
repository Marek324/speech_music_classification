# common.py
# Marek Hric

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

# Unified label mapping used across input_handler, evaluator, frame_label
LABEL_MAP = {"speech": -1, "music": 1, "inactive": 2, "noise": 2}


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
    overlap = {"speech": 0, "music": 0, "inactive": 0}

    for ann in anns:
        for lbl in ["speech", "music", "inactive"]:
            ran = ann.get(lbl)
            if ran is not None:
                start = max(f_start, ran["start"])
                end = min(f_end, ran["end"])
                if start < end:
                    overlap[lbl] += end - start

    if sum(overlap.values()) == 0:
        return LABEL_MAP["inactive"]

    majority = max(overlap.items(), key=lambda x: x[1])[0]
    return LABEL_MAP[majority]


def frame_label_str(
    anns: List[Dict[str, Union[str, int]]], f_start: int, f_end: int
) -> str:
    overlap = {"speech": 0, "music": 0, "inactive": 0}

    for ann in anns:
        lbl = ann.get("label")
        if lbl not in ["speech", "music", "inactive"]:
            raise ValueError(f"Unknown label: {lbl}")

        start = max(f_start, ann["start"])
        end = min(f_end, ann["end"])
        if start < end:
            overlap[lbl] += end - start

    if sum(overlap.values()) == 0:
        return "inactive"

    majority = max(overlap.items(), key=lambda x: x[1])[0]
    return majority


def ms_to_samples(ms: int, sr: int) -> int:
    return int(ms * sr / 1000)
