# main.py
# Marek Hric

import sys
from datasets import IterableDataset, load_dataset, Audio
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print(f"Usage: python {sys.argv[0]}")
    sys.exit(0)


def frame_label(
    anns: List[Dict[str, Optional[Dict[str, int]]]], f_start: int, f_end: int
):
    lbl_val = {"speech": -1, "music": 1, "inactive": 2}
    overlap = {-1: 0, 1: 0, 2: 0}

    for ann in anns:
        for lbl in ["speech", "music", "inactive"]:
            ran = ann.get(lbl)
            if ran is not None:
                start = max(f_start, ran["start"])
                end = min(f_end, ran["end"])
                if start < end:
                    overlap[lbl] += end - start

    majority = max(overlap, key=overlap.get)
    return lbl_val[majority]


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
    metadata: Optional[FrameMetadata]


class SMDataset:
    def __init__(self, s: np.ndarray, m: np.ndarray):
        self.xs: np.ndarray = s
        self.xm: np.ndarray = m
        self.x: np.ndarray = np.vstack((s, m))
        self.t: np.ndarray = np.hstack(
            (np.ones(len(s)).astype(int) * -1, np.ones(len(m)).astype(int))
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self.x[idx], int(self.t[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class InputHandler:
    """
    InputHandler can operate in two modes:
        - "dataset"     → stream HF dataset
        - "microphone"  → real-time mic stream (not implemented, placeholder)
    """

    def __init__(
        self,
        mode: str,
        ds_link: str = "",
        ds_split: str = "train",
        frame_ms: int = 20,
        hop_ms: int = 10,
        sr: int = 16000,
        channels: int = 1,
    ):
        assert mode in ("dataset", "microphone")
        self.mode = mode
        self.sr = sr
        self.channels = channels

        self.frame_len = int(frame_ms * sr / 1000)
        self.hop_len = int(hop_ms * sr / 1000)

        self.frame_start = 0

        self.st_buffer: Optional[np.ndarray] = None

        self.ds_item_class: str = ""
        self.ds_item_subclass: str = ""
        self.ds_item_labels: List[Dict[str, Optional[Dict[str, int]]]] = []

        if self.mode == "dataset":
            assert ds_link != "", "Dataset mode requires ds_link"

            self.dataset = load_dataset(
                ds_link, split=ds_split, streaming=True
            ).cast_column(
                "audio",
                Audio(sampling_rate=self.sr, num_channels=self.channels),
            )

            assert isinstance(self.dataset, IterableDataset)
            self.iterator = iter(self.dataset)

        else:  # microphone mode
            self.iterator = None
            # TODO: Initialize microphone stream here
            # self.mic = pyaudio.PyAudio().open(...)
            self.st_buffer = np.zeros(0, dtype=np.float32)

    def _extract_frame(self) -> np.ndarray:
        # load next item on empty buffer
        if self.st_buffer is None:
            if self.mode == "dataset":
                self._load_next_dataset_item()  # throws EndOfDatasetException
            else:
                self._load_next_mic_buffer()  # TODO: should wait or throw(on end of receiving/error)

        assert self.st_buffer is not None
        available = len(self.st_buffer)

        frame = []

        if self.mode == "dataset":
            if available >= self.frame_len:
                frame = self.st_buffer[: self.frame_len]
                self.st_buffer = self.st_buffer[self.hop_len :]
            else:
                # zero padd the end of dataset item
                frame = np.zeros(self.frame_len, dtype=np.float32)
                frame[:available] = self.st_buffer

                # empty the buffer
                self.st_buffer = None
        else:  # microphone mode
            assert available >= self.frame_len
            frame = self.st_buffer[: self.frame_len]
            self.st_buffer = self.st_buffer[self.hop_len :]

        return frame

    def _load_next_dataset_item(self) -> bool:
        assert self.st_buffer is None
        assert self.iterator is not None

        try:
            item = next(self.iterator)
        except StopIteration:
            raise EndOfDatasetException

        audio = item["audio"].get_all_samples().data
        if hasattr(audio, "cpu"):
            audio.cpu()
        audio = audio.numpy().squeeze()

        self.st_buffer = audio

        # metadata
        self.ds_item_class = item["class"]
        self.ds_item_subclass = item["subclass"]
        self.ds_item_labels = item["labels"]
        self.frame_start = 0

        return True

    def _load_next_mic_buffer(self) -> bool:
        """
        Capture new microphone audio and append to st_buffer.
        Always returns True because microphone is continuous.
        """
        # TODO: Implement microphone reading here
        # raw = self.mic.read(self.hop_len)
        # audio_np = np.frombuffer(raw, dtype=np.float32)
        # self.st_buffer = np.concatenate([self.st_buffer, audio_np])
        return True

    def get_frame(self) -> Optional[FrameData]:
        try:
            audio = self._extract_frame()
        except EndOfDatasetException:
            return None
        # maybe except mic error

        assert audio is not None

        meta = None
        if self.mode == "dataset":
            meta = FrameMetadata(
                label=frame_label(
                    self.ds_item_labels,
                    self.frame_start,
                    self.frame_start + self.frame_len,
                ),
                rec_class=self.ds_item_class,
                subclass=self.ds_item_subclass,
            )

        frame = FrameData(audio=audio, metadata=meta)
        self.frame_start += self.frame_len

        return frame

    def __iter__(self):
        return self

    def __next__(self) -> FrameData:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame


class FrameDataset:
    def __init__(self, input_handler):
        if input_handler.mode != "dataset":
            raise ValueError("FrameDataset requires an InputHandler in dataset mode.")

        print("FrameDataset init")
        self.frames: List[FrameData] = []

        self.classes: Dict[int, List[FrameData]] = {
            -1: [],  # speech
            1: [],  # music
            2: [],  # inactive
        }

        # consume stream
        for frame in input_handler:
            print(frame)
            self._handle_frame(frame)

        self.length = len(self.frames)

    def _handle_frame(self, frame: FrameData):
        self.frames.append(frame)

        meta = frame.metadata
        assert meta is not None

        lbl = meta.label

        self.classes[lbl].append(frame)

    def __len__(self):
        return self.length

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


def main():
    ih = InputHandler(
        "dataset", ds_link="Marek324/speech-music-classification", ds_split="train"
    )
    ds = FrameDataset(ih)
    print(ds.summary)

if __name__=="__main__":
    main()
