from collections import Counter
from typing import Dict, List, Optional, Any, TypedDict

import numpy as np
from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm

import config
from common import (
    FrameData,
    FrameDataStrLabel,
    FrameMetadata,
    FrameMetadataStrLabel,
    frame_label,
    frame_label_str,
)
from feat_extractor import FeatExtractor


class RowStats(TypedDict):
    frames: int
    classes: Counter[str]
    subclasses: Counter[str]


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

        # self.ds_item_class: str = ""
        # self.ds_item_subclass: str = ""
        # self.ds_item_labels: List[Dict[str, Optional[Dict[str, int]]]] = []
        #
        if self.mode == "dataset":
            self.ds_stats = {
                "total": {
                    "frames": 0,
                    "classes": Counter(),
                    "subclasses": Counter(),
                }
            }

            assert ds_link != "", "Dataset mode requires ds_link"

            self.dataset = load_dataset(ds_link, split=ds_split).cast_column(
                "audio",
                Audio(sampling_rate=self.sr, num_channels=self.channels),
            )

            assert isinstance(self.dataset, Dataset)

            processed = self.dataset.map(
                self._process_row,
                desc="Extracting frames/features",
                num_proc=30,
                load_from_cache_file=True,
            )

            feats_all = []
            labels_all = []
            subclasses_all = []

            for row in tqdm(processed, desc="Aggregating"):
                feats_all.extend(row["feats"])
                labels_all.extend(row["labels"])
                subclasses_all.extend(row["subclasses"])

                # rs = row["stats"]
                # self.ds_stats["frames"] += rs["frames"]
                # self.ds_stats["classes"].update(rs["classes"])
                # self.ds_stats["subclasses"].update(rs["subclasses"])
                self.ds_stats["frames"] = len(row["labels"])
                self.ds_stats["classes"].update(row["labels"])
                self.ds_stats["subclasses"].update(row["subclasses"])

            self.X = np.asarray(feats_all, dtype=np.float32)
            self.y = np.asarray(labels_all)
            self.subclasses = np.asarray(subclasses_all, dtype="U20")

            # FINAL SAFETY CHECK
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Final X - Contains Inf: {np.any(np.isinf(self.X))}")
            print(self.X.shape)
            print(self.y.shape)

        else:  # microphone mode
            self.iterator = None
            # TODO: Initialize microphone stream here
            # self.mic = pyaudio.PyAudio().open(...)
            self.st_buffer = np.zeros(0, dtype=np.float32)

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_start: int,
        labels: List[Dict[str, Dict[str, int] | None]],
        _class: str,
        subclass: str,
    ) -> FrameDataStrLabel:
        padd = np.zeros(self.frame_len, dtype=np.float32)
        padd[: len(frame)] = frame
        frame = padd
        feats = self.fextractor.extract(frame)
        meta = FrameMetadataStrLabel(
            label=frame_label_str(
                labels,
                frame_start,
                frame_start + self.frame_len,
            ),
            rec_class=_class,
            subclass=subclass,
        )

        return FrameDataStrLabel(frame, feats, meta)

    def _process_row(self, row) -> Dict[str, Any]:
        audio = row["audio"].get_all_samples().data
        if hasattr(audio, "cpu"):
            audio.cpu()
        audio = audio.numpy().squeeze()

        _class = row["class"]
        subclass = row["subclass"]
        labels = row["labels"]

        feats = []
        labels_list = []
        subclasses = []
        #
        # row_stats: RowStats = {
        #     "frames": 0,
        #     "classes": Counter(),
        #     "subclasses": Counter(),
        # }

        frame_start = 0
        n_samples = len(audio)

        while frame_start + self.frame_len <= n_samples:
            frame = audio[frame_start : frame_start + self.frame_len]

            f = self._process_frame(
                frame,
                frame_start,
                labels,
                _class,
                subclass,
            )

            if np.any(np.isnan(f.feats)) or np.any(np.isinf(f.feats)):
                f.feats = np.zeros_like(f.feats)

            feats.append(f.feats)
            labels_list.append(f.metadata.label)
            subclasses.append(f.metadata.subclass)

            # row_stats["frames"] += 1
            # row_stats["classes"][f.metadata.label] += 1
            # row_stats["subclasses"][f.metadata.subclass] += 1

            frame_start += self.hop_len

        return {
            "feats": feats,
            "labels": labels_list,
            "subclasses": subclasses,
            # "stats": row_stats,
        }

    def getX(self):
        return self.X

    def getY(self):
        return self.y

    def getSubclasses(self):
        return self.subclasses

    def summary(self):
        print("Input Handler summary:")
        if self.mode == "microphone":
            print("mic mode, nothing to sum")
            return

        sr = self.sr
        hop = self.hop_len

        total_frames = self.ds_stats["frames"]
        assert isinstance(total_frames, int)
        total_seconds = (total_frames * hop) / sr
        total_minutes = total_seconds / 60

        print(f"Total Frames:   {total_frames:,}")
        print(f"Total Duration: {total_minutes:.2f} min  ({total_seconds:.2f} sec)")
        print("-" * 60)

        label_names = {-1: "Speech", 1: "Music", 2: "Inactive"}

        print(f"{'CLASS':<25} | {'FRAMES':<10} | {'MINUTES':<10} | {'%':<5}")
        print("-" * 60)

        for lbl, count in self.ds_stats["classes"].items():
            assert isinstance(count, int)
            name = label_names.get(lbl, str(lbl))
            minutes = (count * hop) / sr / 60
            perc = (count / total_frames) * 100
            print(f"{name:<25} | {count:<10,} | {minutes:<10.2f} | {perc:>5.1f}%")

        print("-" * 60)

        print(f"{'SUBCLASS':<30} | {'FRAMES':<10} | {'MINUTES':<10}")
        print("-" * 60)

        for sub, count in self.ds_stats["subclasses"].most_common():
            minutes = (count * hop) / sr / 60
            print(f"{sub:<30} | {count:<10,} | {minutes:<10.2f}")  #

    # def _extract_frame(self) -> np.ndarray:
    #     # load next item on empty buffer
    #     if self.st_buffer is None:
    #         if self.mode == "dataset":
    #             self._load_next_dataset_item()  # throws EndOfDatasetException
    #         else:
    #             self._load_next_mic_buffer()  # TODO: should wait or throw(on end of receiving/error)
    #
    #     assert self.st_buffer is not None
    #     available = len(self.st_buffer)
    #
    #     frame = []
    #
    #     if self.mode == "dataset":
    #         if available >= self.frame_len:
    #             frame = self.st_buffer[: self.frame_len]
    #             self.st_buffer = self.st_buffer[self.hop_len :]
    #         else:
    #             # zero padd the end of dataset item
    #             frame = np.zeros(self.frame_len, dtype=np.float32)
    #             frame[:available] = self.st_buffer
    #
    #             # empty the buffer
    #             self.st_buffer = None
    #     else:  # microphone mode
    #         assert available >= self.frame_len
    #         frame = self.st_buffer[: self.frame_len]
    #         self.st_buffer = self.st_buffer[self.hop_len :]
    #
    #     return frame
    #
    # def _load_next_dataset_item(self) -> bool:
    #     assert self.st_buffer is None
    #     assert self.iterator is not None
    #
    #     try:
    #         item = next(self.iterator)
    #     except StopIteration:
    #         raise EndOfDatasetException
    #
    #     audio = item["audio"].get_all_samples().data
    #     if hasattr(audio, "cpu"):
    #         audio.cpu()
    #     audio = audio.numpy().squeeze()
    #
    #     self.st_buffer = audio
    #
    #     # metadata
    #     self.ds_item_class = item["class"]
    #     self.ds_item_subclass = item["subclass"]
    #     self.ds_item_labels = item["labels"]
    #     self.frame_start = 0
    #
    #     return True
    #
    # def _load_next_mic_buffer(self) -> bool:
    #     """
    #     Capture new microphone audio and append to st_buffer.
    #     Always returns True because microphone is continuous.
    #     """
    #     # TODO: Implement microphone reading here
    #     # raw = self.mic.read(self.hop_len)
    #     # audio_np = np.frombuffer(raw, dtype=np.float32)
    #     # self.st_buffer = np.concatenate([self.st_buffer, audio_np])
    #     return True
    #
    # def get_frame(self) -> Optional[FrameData]:
    #     try:
    #         audio = self._extract_frame()
    #     except EndOfDatasetException:
    #         return None
    #     # maybe except mic error
    #
    #     assert audio is not None
    #
    #     meta = None
    #     if self.mode == "dataset":
    #         meta = FrameMetadata(
    #             label=frame_label(
    #                 self.ds_item_labels,
    #                 self.frame_start,
    #                 self.frame_start + self.frame_len,
    #             ),
    #             rec_class=self.ds_item_class,
    #             subclass=self.ds_item_subclass,
    #         )
    #
    #     frame = FrameData(
    #         audio=audio, feats=self.fextractor.extract(audio), metadata=meta
    #     )
    #     self.frame_start += self.frame_len
    #
    #     return frame
    #
    # def __iter__(self):
    #     return self
    #
    # def __next__(self) -> FrameData:
    #     frame = self.get_frame()
    #     if frame is None:
    #         raise StopIteration
    #     return frame
    #
    # def getX(self) -> np.ndarray:
    #     X = []
    #
    #     # parallelize rows
    #
    #     return np.array(X)
    #
    # def getY(self) -> np.ndarray:
    #     y = []
    #
    #     return np.array(y)
