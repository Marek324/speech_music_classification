# utils.py
# Marek Hric

import sys

try:
    import numpy as np
    import librosa as lb
    import pandas as pd
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import (
    SAMPLE_RATE,
    FRAME_LEN_MS,
    FRAME_HOP_MS,
    REF_PATH,
)


def load_and_resample(file_path:str, target_sr:int = SAMPLE_RATE) -> np.ndarray:
    s = lb.load(file_path, sr=target_sr)[0] # mono
    return s


def framing(s: np.ndarray, fl_ms:int = FRAME_LEN_MS, fh_ms:int = FRAME_HOP_MS, target_sr:int = SAMPLE_RATE) -> np.ndarray:
    fl = int(fl_ms * target_sr / 1000)  # frame length [samples]
    fo = int(fh_ms * target_sr / 1000)  # frame overlap [samples]
    fs = fl - fo  # frame shift [samples]
    Nf = int(1 + np.floor((len(s) - fl) / fs))  # number of frames

    hann_win = np.hanning(fl)

    return np.array([s[i * fs:i * fs + fl] * hann_win for i in range(Nf)])


def load_reference(ref_file: str = REF_PATH, train: bool = True) -> list[dict[str, list[int]]]:
    ref = []
    df = pd.read_csv(ref_file)
    df_filtered = df[df['file'].str.contains('train', na=False)] if train else df[~df['file'].str.contains('train', na=False)]
    for _, row in df_filtered.iterrows():
        ref.append({"file": row['file'], "ref": row['reference']})

    return ref
