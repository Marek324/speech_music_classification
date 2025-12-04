# dataset_builder.py
# Marek Hric

import numpy as np
from pathlib import Path
from collections import deque
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from .dataset import SMDataset
from .data_loader import SMDataLoader
from .segment_statistics import SMSegmentStatistics
from .feature_extractor import SMFeatureExtractor
import o.defaults as defaults


class SMDatasetBuilder:
    def __init__(self):
        _sr = defaults.SAMPLE_RATE
        seg_len = defaults.SEGMENT_LEN_MS * _sr // 1000
        _fl = defaults.FRAME_LEN_MS * _sr // 1000
        _fh = defaults.FRAME_HOP_MS * _sr // 1000
        self._seg_frames: int = int(np.floor((seg_len - _fl) / _fh) + 1)

        self._loader: SMDataLoader = SMDataLoader(
            path=Path(defaults.DATASET_PATH).absolute(), sr=_sr, fl=_fl, fh=_fh
        )
        print("SMDatasetBuilder created")

    def build(self, train: bool) -> SMDataset:
        recs: list[tuple[list[int], list[np.ndarray]]] = self._loader.load(train=train)

        process_fn = partial(
            self._process_recording,
            sr=defaults.SAMPLE_RATE,
            seg_frames=self._seg_frames,
        )

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(process_fn, recs),
                    total=len(recs),
                    desc="Building dataset",
                )
            )

        speech_list: list[np.ndarray] = []
        music_list: list[np.ndarray] = []

        for speech_feats, music_feats in results:
            speech_list.extend(speech_feats)
            music_list.extend(music_feats)

        return SMDataset(X_speech=np.vstack(speech_list), X_music=np.vstack(music_list))

    def _process_recording(
        self, rec: np.ndarray, sr: int, seg_frames: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        ex = SMFeatureExtractor(sr)
        st = SMSegmentStatistics(seg_frames)

        ref, frames = rec
        win = deque(maxlen=seg_frames)
        speech_list: list[np.ndarray] = []
        music_list: list[np.ndarray] = []

        for frame, ref_c in zip(frames, ref):
            feats = ex.extract(frame)
            win.append(feats)

            if ref_c == 2:  # silence
                continue

            stats = st.compute(np.array(list(win)))
            feats = np.hstack((feats, stats))

            match ref_c:
                case -1:  # speech
                    speech_list.append(feats)
                case 1:  # music
                    music_list.append(feats)

        return speech_list, music_list
