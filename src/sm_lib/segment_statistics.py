# segment_statistis.py
# Marek Hric

import numpy as np
from scipy.stats import skew

class SMSegmentStatistics:
    def __init__(self, seg_frames: int):
        self.segment_frames: int= seg_frames

    def compute(self, feats: np.ndarray) -> np.ndarray:
        n_frames = feats.shape[0]
        assert n_frames <= self.segment_frames
        padd = self.segment_frames - n_frames

        feats = np.vstack((
            np.zeros((padd, feats.shape[1])),
            feats
        ))

        diffs = np.abs(np.diff(feats, axis=0))
        return np.hstack([
            np.mean(feats, axis=0),
            np.std(feats, axis=0),
            np.mean(diffs, axis=0),
            np.std(diffs, axis=0),
            np.nan_to_num(skew(feats[:, 1]), nan=0.0),
            np.nan_to_num(skew(diffs[:, 1]), nan=0.0),
            self._lster(feats[:, 0])
        ])


    def _lster(self, energies: np.ndarray) -> float:
        thr = float(np.mean(energies) / 3)
        return np.sum(energies < thr) / len(energies)

