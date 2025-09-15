# main.py
# Marek Hric

import sys
import os

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print(f"Usage: python {sys.argv[0]}")
    sys.exit(0)

try:
    import numpy as np
    import scipy.signal as sg
    import librosa as lb
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import *
import features as ft
from utils import *

def main():
    train("speech")
    evaluate()


# dir = {speech, music}
def train(dir:str):
    prefix = DATASET_PATH + f"/train/{dir}"
    for file in os.listdir(prefix):
        print(f"Processing {file}...")
        file_path = f"{prefix}/{file}"
        s = load_and_resample(file_path, SAMPLE_RATE)
        features = create_features(s)
        break


def create_features(s: np.ndarray) -> np.ndarray:
    features = np.array([])
    frames = framing(s)
    f_prev = None

    # debug counter
    # counter = 2

    for f in frames:
        frame_feat = ft.short_time_energy(f)
        frame_feat = np.append(frame_feat, ft.zero_crossing_rate(f))
        frame_feat = np.append(frame_feat, ft.band_energy_ratio(f, 0, 70, 11000, 44100))
        frame_feat = np.append(frame_feat, ft.autocorrelation_coeff(f))
        frame_feat = np.hstack((frame_feat, ft.mfcc(f).flatten()))
        frame_feat = np.append(frame_feat, ft.spectrum_rolloff_point(f))
        frame_feat = np.append(frame_feat, ft.spectrum_centroid(f))
        frame_feat = np.append(frame_feat, ft.spectrum_spread(f))

    
        if f_prev is not None:
            frame_feat = np.append(
                frame_feat,
                ft.mfcc_diff_norm(ft.mfcc(f).flatten(), ft.mfcc(f_prev).flatten())
            )
            frame_feat = np.append(frame_feat, ft.spectral_flux(f, f_prev))
        else:
            frame_feat = np.append(frame_feat, 0)  # mfcc diff norm
            frame_feat = np.append(frame_feat, 0)  # spectral flux

        features = np.vstack((features, frame_feat)) if features.size else frame_feat.reshape(1, -1)

        #print(f"features shape: {features.shape}")
        f_prev = f

        # debug counter
     #   counter -= 1
     #   if counter == 0:
     #       break

    return features


def evaluate():
    ...


if __name__ == "__main__":
    main()
