# time_domain.py
# Marek Hric

import numpy as np


def short_time_energy(frame: np.ndarray) -> float:
    return 10 * np.log10(1/frame.shape[0] * np.sum(frame**2))


def zero_crossing_rate(frame: np.ndarray) -> float:
    count = 0
    for i in range(1, frame.shape[0]):
        if frame[i] * frame[i - 1] < 0:
            count += 1
    return count / 2


def band_energy_ratio(
        frame: np.ndarray, 
        fl_1: float, fh_1: float, 
        fl_2: float, fh_2: float,
        sr: int, n_fft
    ) -> float:

    def bin_num(f: float, sr: int, K: int) -> int:
        return int(np.floor((K * f) / sr))

    
    def band_energy(dft: np.ndarray, fl: float, fh: float) -> float:
        K = dft.shape[0]
        b1 = bin_num(fl, sr, K)
        b2 = bin_num(fh, sr, K)
        return np.sum(np.abs(dft[b1:b2])**2)


    dft = np.fft.fft(frame, n=n_fft)
    E1 = band_energy(dft, fl_1, fh_1)
    E2 = band_energy(dft, fl_2, fh_2)

    return 10 * np.log10(E1 / E2) if E2 > 0 else np.inf


def autocorrelation_coeff(frame: np.ndarray, sr:int)-> float:
    R = np.correlate(frame, frame, mode='full')
    R = R[R.shape[0] // 2:] 
    m1 = np.floor(3 * sr / 1000).astype(int)  # 3ms
    m2 = np.floor(16 * sr / 1000).astype(int) # 16ms
    return np.max(R[m1:m2]) 


