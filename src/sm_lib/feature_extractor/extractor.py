# extractor.py
# Marek Hric

import numpy as np
from copy import deepcopy

from .time_domain import (
    short_time_energy,
    zero_crossing_rate,
    band_energy_ratio,
    autocorrelation_coeff,
)

from .spectrum import (
    spectrum_rolloff_point,
    spectrum_centroid,
    spectrum_spread,
    spectral_flux,
)

from .mfcc import (
    mfcc,
    mfcc_diff_norm
)


class SMFeatureExtractor:
    def __init__(self, sr: int, n_fft:int = 1024):
        self.SAMPLE_RATE = sr # import from default in caller
        self.N_FFT = n_fft

        self.last_frame = None
        self.last_mfcc = None
        print("SMFeatureExtractor created")


    def extract(self, frame: np.ndarray) -> np.ndarray:
        SR = self.SAMPLE_RATE
        NF = self.N_FFT
        res =  np.hstack((
            (curr_mfcc := mfcc(frame, SR, NF).flatten()),
            np.array([
                short_time_energy(frame),
                zero_crossing_rate(frame),
                band_energy_ratio(frame, 0, 70, 11000, 44100, SR, NF),
                autocorrelation_coeff(frame, SR),
                spectrum_rolloff_point(frame, SR, NF),
                spectrum_centroid(frame, SR, NF),
                spectrum_spread(frame, SR, NF),
                spectral_flux(frame, self.last_frame) if self.last_frame is not None else 0,
                mfcc_diff_norm(curr_mfcc, self.last_mfcc) if self.last_mfcc is not None else 0
            ])
        ))

        self.last_frame = deepcopy(frame)
        self.last_mfcc = deepcopy(curr_mfcc)
        return res

    
    def reset(self):
        self.last_frame = None
        self.last_mfcc = None
