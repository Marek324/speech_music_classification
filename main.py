# main.py
# Marek Hric

import sys

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print("Usage: python main.py")
    sys.exit(0)

try:
    import os
    import numpy as np
    import scipy.signal as sg
    import librosa as lb
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

SAMPLE_RATE = 44100  # Default sample rate
FRAME_LEN_MS = 40  # Frame length in milliseconds
FRAME_HOP_MS = 20  # Frame hop in milliseconds
DATASET_PATH = "dataset/music-speech/wavfile"  # Path to the dataset


def main():
    train()
    evaluate()

# dir = {speech, music}
def train(dir:str):
    prefix = DATASET_PATH + f"/train/{dir}"
    for file in os.listdir(prefix):
        file_path = f"{prefix}/{file}"
        s = load_and_resample(file_path, SAMPLE_RATE)
        features = create_features(s)

def create_features(s: np.ndarray) -> np.ndarray:
    features = np.array([])
    frames = framing(s)
    for f in frames:
        ...

    return features

def is_silence(frame: np.ndarray, threshold: int = -20) -> bool:
    # Normalized short time energy
    E = np.sum(np.abs(np.fft.rfft(a=frame, n=len(frame)))**2) / len(frame)**2
    db = 10 * np.log10(E) if E > 0 else -np.inf
    return db < threshold

def framing(s: np.ndarray, fl_ms:int = FRAME_LEN_MS, fh_ms:int = FRAME_HOP_MS) -> np.ndarray:
    fl = int(fl_ms * SAMPLE_RATE / 1000)  # frame length [samples]
    fo = int(fh_ms * SAMPLE_RATE / 1000)  # frame overlap [samples]
    fs = fl - fo  # frame shift [samples]
    Nf = int(1 + np.floor((len(s) - fl) / fs))  # number of frames

    hann_win = np.hanning(fl)

    return np.array([s[i * fs:i * fs + fl] * hann_win for i in range(Nf)])

def evaluate():
    ...

def load_and_resample(file_path:str, target_fs:int) -> np.ndarray:
    s = lb.load(file_path, sr=target_fs)[0] # mono only for now
    return s

if __name__ == "__main__":
    main()
