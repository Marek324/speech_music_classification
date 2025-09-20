# frame_labeling.py
# Marek Hric

def help_exit(exit_code: int = 0):
    print("Usage: python frame_labeling.py <dataset_dir>")
    sys.exit(exit_code)

import sys

if "-h" in sys.argv or "--help" in sys.argv:
    help_exit()
if len(sys.argv) != 2:
    help_exit(1)

try:
    import pandas as pd
    import numpy as np
    import librosa as lb
    import os
    import re
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)


# defaults
SAMPLE_RATE = 44100
FRAME_LEN_MS = 40
FRAME_HOP_MS = 20


def framing(s: np.ndarray, fl_ms:int = FRAME_LEN_MS, fh_ms:int = FRAME_HOP_MS, target_sr:int = SAMPLE_RATE) -> np.ndarray:
    fl = int(fl_ms * target_sr / 1000)  # frame length [samples]
    fo = int(fh_ms * target_sr / 1000)  # frame overlap [samples]
    fs = fl - fo  # frame shift [samples]
    Nf = int(1 + np.floor((len(s) - fl) / fs))  # number of frames

    hann_win = np.hanning(fl)

    return np.array([s[i * fs:i * fs + fl] * hann_win for i in range(Nf)])


def load_and_resample(file_path:str, target_sr:int = SAMPLE_RATE) -> np.ndarray:
    s = lb.load(file_path, sr=target_sr)[0] # mono
    return s


# other left out
SPEECH_DIRS = ['test/speech', 'train/speech', 'train/m+s']
MUSIC_DIRS = ['test/music/novocals', 'test/music/vocals', 'train/music']

dir = sys.argv[1]

def sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def is_silence(frame:np.ndarray, thr:float) -> int:
    E = np.sum(frame**2)
    return 1 if E < thr else 0

def create_reference(sig:np.ndarray, thr:float, not_sil_ref:int) -> dict[str, np.ndarray]:
    ref = []
    frames = framing(sig)
    for f in frames:
        if is_silence(f, thr):
            ref.append(0)
        else:
            ref.append(not_sil_ref)

    return ref


data = []
for d in os.walk(dir):
    # leaf dir
    if len(d[1]) == 0:
        path = f"{d[0]}"
        rel_path = os.path.relpath(path, dir)
        if rel_path not in SPEECH_DIRS and rel_path not in MUSIC_DIRS:
            print(f"Skipping {rel_path}...")
            continue
        ref_val = 1 if rel_path in SPEECH_DIRS else 2
        for wav in sorted(d[2], key=sort_key):
            wav_path = os.path.join(path, wav)  
            sig = load_and_resample(wav_path)
            thr = 0.2
            ref = create_reference(sig, thr, ref_val)

            data.append({"file": f"{rel_path}/{wav}", "reference": ref})


pd.DataFrame(data).to_csv(f"{dir}/reference.csv", index=False)
