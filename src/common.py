from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, List


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


def frame_label(
    anns: List[Dict[str, Optional[Dict[str, int]]]], f_start: int, f_end: int
):
    lbl_val = {"speech": -1, "music": 1, "inactive": 2}
    overlap = {"speech": 0, "music": 0, "inactive": 0}

    for ann in anns:
        for lbl in ["speech", "music", "inactive"]:
            ran = ann.get(lbl)
            if ran is not None:
                start = max(f_start, ran["start"])
                end = min(f_end, ran["end"])
                if start < end:
                    overlap[lbl] += end - start

    majority = max(overlap.items(), key=lambda x: x[1])[0]
    return lbl_val[majority]
