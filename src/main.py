# main.py
# Marek Hric

import sys

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print(f"Usage: python {sys.argv[0]}")
    sys.exit(0)


try:
    import numpy as np
    from scipy.stats import skew
    from collections import deque
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import (
    DATASET_PATH,
    FRAME_LEN_MS,
    FRAME_HOP_MS,
    SEGMENT_LEN_MS,
)

from utils import (
    load_and_resample,
    framing,
    load_reference,
    seg_frame_count
)

import features as ft
from decision_tree import DecisionTree


def main():
    classifier = DecisionTree().train(
        *create_features(load_reference())
    )

def create_features(ref: list[dict[str, list[float]]]) -> tuple[np.ndarray, np.ndarray]:

    def extract_features(
        frame: np.ndarray,
        frame_prev: np.ndarray = None,
        frame_mfcc_prev: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.hstack((
            (frame_mfcc :=ft.mfcc(frame).flatten()),
            np.array([
                ft.short_time_energy(frame),
                ft.zero_crossing_rate(frame),
                ft.band_energy_ratio(frame, 0, 70, 11000, 44100),
                ft.autocorrelation_coeff(frame),
                ft.spectrum_rolloff_point(frame),
                ft.spectrum_centroid(frame),
                ft.spectrum_spread(frame),
                ft.spectral_flux(frame, frame_prev) if frame_prev is not None else 0,
                ft.mfcc_diff_norm(frame_mfcc, frame_mfcc_prev) if frame_mfcc_prev is not None else 0
            ])
        )), frame_mfcc

    def feature_statistics(
        features: np.ndarray,
        seg_len_ms: int = SEGMENT_LEN_MS,
        f_len_ms: int = FRAME_LEN_MS,
       f_hop_ms: int = FRAME_HOP_MS
    ) -> np.ndarray:

        req_frames = seg_frame_count()

        n_feats = features.shape[1]

        padd = req_frames - features.shape[0]
        features = np.vstack((
            np.zeros((padd, n_feats)),
            features
        ))

        diffs = np.abs(np.diff(features, axis=0))
        return np.hstack([
            np.mean(features, axis=0),
            np.std(features, axis=0),
            np.mean(diffs, axis=0),
            np.std(diffs, axis=0),
            np.nan_to_num(skew(features[:, 1]), nan=0.0),
            np.nan_to_num(skew(diffs[:, 1]), nan=0.0),
            ft.lster(features[:, 0])
        ])

    # === create_features() ===
    speech_list = []
    music_list = []

    for entry in ref:
        s = load_and_resample(f"{DATASET_PATH}/{entry['file']}")
        frames = framing(s)

        last_frame = None
        last_frame_mfcc = None
        stat_win = deque(maxlen=seg_frame_count())

        for frame, ref_c in zip(frames, entry['ref']):
            feats, last_frame_mfcc = extract_features(frame, last_frame, last_frame_mfcc)
            stat_win.append(feats)
            stats = feature_statistics(np.array(list(stat_win)))
            feats = np.hstack((feats, stats))
            match ref_c:
                case 0: # silence
                    continue
                case 1: # speech
                    speech_list.append(feats)
                case 2: # music
                    music_list.append(feats)

            last_frame = frame

    X_speech = np.vstack(speech_list)
    X_music = np.vstack(music_list)

    return X_speech, X_music


if __name__ == "__main__":
    main()
