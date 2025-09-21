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
    from sklearn import tree
    from ast import literal_eval
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import DATASET_PATH

from utils import (
    load_and_resample,
    framing,
    load_reference
)

import features as ft

def main():
    classifier = tree.DecisionTreeClassifier()
    train(classifier)
    evaluate(classifier)
 

def train(clf: tree.DecisionTreeClassifier):
    ref_dict = load_reference()
    ref = np.array([])
    features = np.array([])
    for entry in ref_dict:
        file_path = f"{DATASET_PATH}/{entry['file']}"
        s = load_and_resample(file_path)
        features = np.vstack((features, create_features(s))) if features.size else create_features(s)
        ref = np.hstack((ref, np.array(literal_eval(entry['ref'])))) if ref.size else np.array(literal_eval(entry['ref']))

    print(f"Training on {features.shape[0]} frames with {features.shape[1]} features each.")
    clf.fit(features, ref)
    print("Training complete.")


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
            frame_feat = np.append(frame_feat,
                ft.mfcc_diff_norm(ft.mfcc(f).flatten(), ft.mfcc(f_prev).flatten()))
            frame_feat = np.append(frame_feat, ft.spectral_flux(f, f_prev))
        else:
            frame_feat = np.append(frame_feat, 0)  # mfcc diff norm
            frame_feat = np.append(frame_feat, 0)  # spectral flux

        features = np.vstack((features, frame_feat)) if features.size else frame_feat.reshape(1, -1)

        f_prev = f

        # debug counter
     #   counter -= 1
     #   if counter == 0:
     #       break

    return features


def evaluate(clf: tree.DecisionTreeClassifier):
    ref = load_reference(train=False)
    counter = 0
    for entry in ref:
        file_path = f"{DATASET_PATH}/{entry['file']}"
        s = load_and_resample(file_path)
        features = create_features(s)
        print(f"Evaluating {entry['file']} on {features.shape[0]} frames with {features.shape[1]} features each.")
        res = clf.predict(features)
        if np.array_equal(res, np.array(literal_eval(entry['ref']))):
            counter += 1

    print(f"Evaluation complete. {counter}/{len(ref)} files classified correctly.")

if __name__ == "__main__":
    main()
