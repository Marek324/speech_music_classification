# dataset_builder.py
# Marek Hric

import numpy as np
from pathlib import Path
from collections import deque
from tqdm import tqdm

from .dataset import SMDataset
from .data_loader import SMDataLoader
from .segment_statistics import SMSegmentStatistics
from .feature_extractor import SMFeatureExtractor
import defaults


DEBUG = True
DEBUG_FRAMES = True


class SMDatasetBuilder:
    def __init__(self):
        _sr: int = defaults.SAMPLE_RATE
        seg_len = defaults.SEGMENT_LEN_MS * _sr // 1000
        _fl = defaults.FRAME_LEN_MS * _sr // 1000
        _fh = defaults.FRAME_HOP_MS * _sr // 1000
        self._seg_frames: int = int(np.floor((seg_len - _fl) / _fh) + 1)

        self._loader: SMDataLoader = SMDataLoader(
            path=Path(defaults.DATASET_PATH).absolute(),
            sr=_sr,
            fl=_fl,
            fh=_fh
        )
        self._extractor: SMFeatureExtractor = SMFeatureExtractor(defaults.SAMPLE_RATE)
        self._statistics: SMSegmentStatistics = SMSegmentStatistics(self._seg_frames)
        print("SMDatasetBuilder created")


    def build(self, train: bool) -> SMDataset:
        ex: SMFeatureExtractor = self._extractor
        st: SMSegmentStatistics = self._statistics

        speech_list: list[np.ndarray] = []
        music_list: list[np.ndarray] = []

        recs: list[tuple[list[int], list[np.ndarray]]] = self._loader.load(train=train)
        for ref, frames in tqdm(recs, desc="Building dataset"):
            if DEBUG:
                if any([r == 1 for r in ref]) and len(speech_list) >= DEBUG_FRAMES: continue # noqa: E701
                elif any([r == 2 for r in ref]) and len(music_list) >= DEBUG_FRAMES: continue # noqa: E701


            win = deque(maxlen=self._seg_frames)
            for frame, ref_c in zip(frames, ref):
                feats = ex.extract(frame)
                win.append(feats)

                if ref_c == 0:
                    continue

                stats = st.compute(np.array(list(win)))
                feats = np.hstack((feats, stats))

                match ref_c:
                    case 1: # speech
                        speech_list.append(feats)
                    case 2: # music
                        music_list.append(feats)

            ex.reset() # reset extractor memory for new file

        return SMDataset(
            X_speech = np.vstack(speech_list),
            X_music = np.vstack(music_list)
        )
        
