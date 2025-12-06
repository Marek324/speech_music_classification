from datasets import IterableDataset, load_dataset, Audio
from typing import Optional, Dict, List
import numpy as np
from common import EndOfDatasetException, FrameData, FrameMetadata, frame_label
from feat_extractor import FeatExtractor
import config


class InputHandler:
    """
    InputHandler can operate in two modes:
        - "dataset"     → stream HF dataset
        - "microphone"  → real-time mic stream (not implemented, placeholder)
    """

    def __init__(
        self,
        mode: str,
        feat_extractor: FeatExtractor,
        ds_link: str = "",
        ds_split: str = "train",
    ):
        assert mode in ("dataset", "microphone")
        self.mode = mode
        self.fextractor = feat_extractor

        defaults = config.get_config().defaults
        self.sr = defaults.sample_rate
        self.channels = defaults.channels

        self.frame_len = defaults.frame_length
        self.hop_len = defaults.hop_length

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

        frame = FrameData(
            audio=audio, feats=self.fextractor.extract(audio), metadata=meta
        )
        self.frame_start += self.frame_len

        return frame

    def __iter__(self):
        return self

    def __next__(self) -> FrameData:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame
