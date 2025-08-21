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
    import soundfile as sf
    import librosa
    import os
    import re
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

SAMPLE_RATE = 44100  # Default sample rate
SPEECH_DIRS = ['test/speech', 'train/speech'] # m+s and other left out for now
MUSIC_DIRS = ['test/music/novocals', 'train/music/vocals', 'train/music']

dir = sys.argv[1]

# default values, also used in classifier
fl = 20 * SAMPLE_RATE // 1000
fh = 10 * SAMPLE_RATE // 1000
thr = 0.2 

def sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]

def framing(sig:np.ndarray, fl:int, fh:int):
    Nf = int(1 + np.floor((len(sig) - fl) / fh)) # number of frames

    hann_win = np.hanning(fl)

    return np.array([sig[i * fh:i * fh + fl] * hann_win for i in range(Nf)])

def is_silence(frame:np.ndarray, thr:float) -> int:
    E = np.sum(frame**2)
    return 1 if E < thr else 0

def create_reference(sig:np.ndarray, fl:int, fh:int, thr:float, not_sil_ref:int) -> dict[str, np.ndarray]:
    ref = []
    for f in framing(sig, fl, fh):
        if is_silence(f, thr):
            ref.append(0)
        else:
            ref.append(not_sil_ref)

    return ref


def thr_test(sig:np.ndarray, ref:list[int], fl:int, fh:int):
    frames = framing(sig, fl, fh)

    non_silent = []
    for i, frame in enumerate(frames):
        if ref[i] == 1:
            non_silent.append(frame)

    non_silent = np.array(non_silent)

    print(sig.shape, frames.shape, non_silent.shape)

    N = len(non_silent) * fh + fl
    out = np.zeros((N))
    for i, f in enumerate(non_silent):
        out[i*fh:i*fh+fl] = f

    sf.write("test.wav", out, SAMPLE_RATE)

data = []
skip = 24
for d in os.walk(dir):
    # leaf dir
    if len(d[1]) == 0:
        path = f"{d[0]}"
        rel_path = os.path.relpath(path, dir)
        if rel_path not in SPEECH_DIRS and rel_path not in MUSIC_DIRS:
            continue
        ref_val = 1 if rel_path in SPEECH_DIRS else 2
        for wav in sorted(d[2], key=sort_key):
            skip -= 1
            if skip > 0:
                continue
            wav_path = os.path.join(path, wav)  
            print(wav_path)
            sig, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
            ref = create_reference(sig, fl, fh, thr, ref_val)
            # thr_test(sig, ref, fl, fh)
            data.append({"file": d[0], "reference": ref})

out_dir = os.path.dirname(dir.rstrip(os.sep))

pd.DataFrame(data).to_csv(f"{out_dir}/reference.csv", index=False)
