# mfcc.py
# Marek Hric

import numpy as np
from librosa import feature as ft

def mfcc(frame: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    return ft.mfcc(y=frame, sr=sr, n_mfcc=10, hop_length=len(frame)+1, n_fft=n_fft)


def mfcc_diff_norm(mfccs: np.ndarray, mfccs_prev: np.ndarray) -> float:
    return np.sqrt(np.sum(np.abs(mfccs - mfccs_prev)**2))

