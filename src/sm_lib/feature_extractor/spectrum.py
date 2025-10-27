# spectrum.py
# Marek Hric

import numpy as np
from librosa import feature as ft


def spectrum_rolloff_point(
    frame: np.ndarray, sr: int, n_fft: int, thr: float = 0.85
) -> float:
    return float(
        ft.spectral_rolloff(y=frame, sr=sr, roll_percent=thr, n_fft=n_fft)[0, 0]
    )


def spectrum_centroid(frame: np.ndarray, sr: int, n_fft: int) -> float:
    return float(ft.spectral_centroid(y=frame, sr=sr, n_fft=n_fft)[0, 0])


def spectrum_spread(frame: np.ndarray, sr: int, n_fft: int) -> float:
    return float(ft.spectral_bandwidth(y=frame, sr=sr, n_fft=n_fft)[0, 0])


def spectral_flux(frame: np.ndarray, frame_prev: np.ndarray) -> float:
    return np.sum(np.abs(np.fft.fft(frame) - np.fft.fft(frame_prev)) ** 2)
