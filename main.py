# main.py
# Marek Hric

import sys

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print(f"Usage: python {sys.argv[0]}")
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


def short_time_energy(frame: np.ndarray) -> float:
    return 10 * np.log10(1/frame.shape[0] * np.sum(frame**2))


def zero_crossing_rate(frame: np.ndarray) -> float:
    count = 0
    for i in range(1, frame.shape[0]):
        if frame[i] * frame[i - 1] < 0:
            count += 1
    return count / 2


# 0-70, 11k-44k1
def band_energy_ratio(frame: np.ndarray, fl_1: float, fh_1: float, fl_2: float, fh_2: float) -> float:
    def bin_num(f: float, Fs: int, K: int) -> int:
        return int(np.floor((K * f) / Fs))

    
    def band_energy(dft: np.ndarray, fl: float, fh: float) -> float:
        global SAMPLE_RATE
        K = dft.shape[0]
        b1 = bin_num(fl, SAMPLE_RATE, K)
        b2 = bin_num(fh, SAMPLE_RATE, K)
        return np.sum(np.abs(dft[b1:b2])**2)


    dft = np.fft.fft(frame)
    E1 = band_energy(dft, fl_1, fh_1)
    E2 = band_energy(dft, fl_2, fh_2)
    return 10 * np.log10(E1 / E2) if E2 > 0 else np.inf


def autocorrelation_coeff(frame: np.ndarray)-> float:
    R = np.correlate(frame, frame, mode='full')
    R = R[R.shape[0] // 2:] 
    m1 = np.floor(3 * SAMPLE_RATE / 1000).astype(int)  # 3ms
    m2 = np.floor(16 * SAMPLE_RATE / 1000).astype(int) # 16ms
    return np.max(R[m1:m2]) 


def mfcc(frame: np.ndarray)-> np.ndarray:
    return lb.feature.mfcc(y=frame, sr=SAMPLE_RATE, n_mfcc=10)


def mfcc_diff_norm(mfccs: np.ndarray, mfccs_prev: np.ndarray) -> float:
    return np.sqrt(np.sum(np.abs(mfccs - mfccs_prev)**2))


def spectrum_rolloff_point(frame: np.ndarray, thr: float = 0.85) -> float:
    return lb.feature.spectral_rolloff(y=frame, sr=SAMPLE_RATE, roll_percent=thr)[0, 0]


def spectrum_centroid(frame: np.ndarray) -> float:
    return lb.feature.spectral_centroid(y=frame, sr=SAMPLE_RATE)[0, 0]


def spectral_flux(frame: np.ndarray, frame_prev: np.ndarray) -> float:
    return np.sum(np.abs(np.fft.fft(frame) - np.fft.fft(frame_prev))**2)


def spectrum_spread(frame: np.ndarray) -> float:
    return lb.feature.spectral_bandwidth(y=frame, sr=SAMPLE_RATE)[0, 0]


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
