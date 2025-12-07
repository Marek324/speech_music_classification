from typing import Dict, List
import numpy as np
from common import FrameData


class FrameDataset:
    def __init__(self, input_handler):
        """
        frames:  List[FrameData]
        classes: Dict[int, List[FrameData]]
        """
        if input_handler.mode != "dataset":
            raise ValueError("FrameDataset requires an InputHandler in dataset mode.")

        self.frames: List[FrameData] = []

        self.classes: Dict[int, List[FrameData]] = {
            -1: [],  # speech
            1: [],  # music
            2: [],  # inactive
        }

        # consume stream
        for frame in input_handler:
            self._handle_frame(frame)

    def _handle_frame(self, frame: FrameData):
        self.frames.append(frame)

        meta = frame.metadata
        assert meta is not None

        lbl = meta.label

        self.classes[lbl].append(frame)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index) -> FrameData:
        return self.frames[index]

    def get_all_audio(self) -> np.ndarray:
        return np.stack([f.audio for f in self.frames])

    def get_all_labels(self) -> List[str]:
        labels = []
        for f in self.frames:
            if f.metadata is None:
                labels.append("unknown")
            else:
                labels.append(f.metadata.label)
        return labels

    def get_all_subclasses(self) -> List[str]:
        subclasses = []
        for f in self.frames:
            if f.metadata is None:
                subclasses.append("unknown")
            else:
                subclasses.append(f.metadata.subclass)
        return subclasses

    def summary(self) -> str:
        msg = [
            f"Total frames: {len(self.frames)}",
            f"  speech:    {len(self.classes[-1])}",
            f"  music:     {len(self.classes[1])}",
            f"  inactive:  {len(self.classes[2])}",
        ]

        return "\n".join(msg)
