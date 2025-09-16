# features.py
# Marek Hric

import sys

try:
    import numpy as np
    import librosa as lb
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import SAMPLE_RATE


def short_time_energy(frame: np.ndarray) -> float:
    return 10 * np.log10(1/frame.shape[0] * np.sum(frame**2))


def zero_crossing_rate(frame: np.ndarray) -> float:
    count = 0
    for i in range(1, frame.shape[0]):
        if frame[i] * frame[i - 1] < 0:
            count += 1
    return count / 2


# 0-70, 11k-44k1
def band_energy_ratio(
        frame: np.ndarray, 
        fl_1: float, fh_1: float, 
        fl_2: float, fh_2: float,
        sr: int = SAMPLE_RATE
    ) -> float:

    def bin_num(f: float, Fs: int, K: int) -> int:
        return int(np.floor((K * f) / Fs))

    
    def band_energy(dft: np.ndarray, fl: float, fh: float) -> float:
        K = dft.shape[0]
        b1 = bin_num(fl, sr, K)
        b2 = bin_num(fh, sr, K)
        return np.sum(np.abs(dft[b1:b2])**2)


    dft = np.fft.fft(frame)
    E1 = band_energy(dft, fl_1, fh_1)
    E2 = band_energy(dft, fl_2, fh_2)

    return 10 * np.log10(E1 / E2) if E2 > 0 else np.inf


def autocorrelation_coeff(frame: np.ndarray, sr:int = SAMPLE_RATE)-> float:
    R = np.correlate(frame, frame, mode='full')
    R = R[R.shape[0] // 2:] 
    m1 = np.floor(3 * sr / 1000).astype(int)  # 3ms
    m2 = np.floor(16 * sr / 1000).astype(int) # 16ms
    return np.max(R[m1:m2]) 


def mfcc(frame: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    return lb.feature.mfcc(y=frame, sr=sr, n_mfcc=10, hop_length=len(frame)+1, n_fft=1024)


def mfcc_diff_norm(mfccs: np.ndarray, mfccs_prev: np.ndarray) -> float:
    return np.sqrt(np.sum(np.abs(mfccs - mfccs_prev)**2))


def spectrum_rolloff_point(frame: np.ndarray, thr: float = 0.85, sr: int = SAMPLE_RATE) -> float:
    return lb.feature.spectral_rolloff(y=frame, sr=sr, roll_percent=thr, n_fft=1024)[0, 0]


def spectrum_centroid(frame: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    return lb.feature.spectral_centroid(y=frame, sr=sr, n_fft=1024)[0, 0]


def spectral_flux(frame: np.ndarray, frame_prev: np.ndarray) -> float:
    return np.sum(np.abs(np.fft.fft(frame) - np.fft.fft(frame_prev))**2)


def spectrum_spread(frame: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    return lb.feature.spectral_bandwidth(y=frame, sr=sr, n_fft=1024)[0, 0]
