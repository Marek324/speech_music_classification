# utils.py
# Marek Hric

import sys

try:
    import numpy as np
    import scipy.signal as sg
    import librosa as lb
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import *


def load_and_resample(file_path:str, target_fs:int) -> np.ndarray:
    s = lb.load(file_path, sr=target_fs)[0] # mono
    return s


def framing(s: np.ndarray, fl_ms:int = FRAME_LEN_MS, fh_ms:int = FRAME_HOP_MS) -> np.ndarray:
    fl = int(fl_ms * SAMPLE_RATE / 1000)  # frame length [samples]
    fo = int(fh_ms * SAMPLE_RATE / 1000)  # frame overlap [samples]
    fs = fl - fo  # frame shift [samples]
    Nf = int(1 + np.floor((len(s) - fl) / fs))  # number of frames

    hann_win = np.hanning(fl)

    return np.array([s[i * fs:i * fs + fl] * hann_win for i in range(Nf)])
