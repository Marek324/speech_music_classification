# main.py
# Marek Hric

import sys

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print(f"Usage: python {sys.argv[0]}")
    sys.exit(0)

try:
    import numpy as np
    from sklearn import tree
    from ast import literal_eval
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    from scipy.stats import skew
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import (
    DATASET_PATH,
    FRAME_LEN_MS,
    SEGMENT_LEN_MS
)

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
    frames = framing(s)
    features_list = []
    array_list = []

    seg_len = SEGMENT_LEN_MS // FRAME_LEN_MS
    seg_counter = 0

    f_prev = None
    f_mfcc_prev = None

    for i, f in enumerate(frames):
        f_mfcc = ft.mfcc(f).flatten()
        f_feats_scalars = np.array([
            ft.short_time_energy(f),
            ft.zero_crossing_rate(f),
            ft.band_energy_ratio(f, 0, 70, 11000, 44100),
            ft.autocorrelation_coeff(f),
            ft.spectrum_rolloff_point(f),
            ft.spectrum_centroid(f),
            ft.spectrum_spread(f),
            ft.spectral_flux(f, f_prev) if f_prev is not None else 0,
            ft.mfcc_diff_norm(f_mfcc, f_mfcc_prev) if f_mfcc_prev is not None else 0
        ])

        f_prev = f
        f_mfcc_prev = f_mfcc

        features = np.array(np.hstack((f_feats_scalars, f_mfcc)))
        features_list.append(features)

        seg_counter += 1
        if (i + 1) % seg_len == 0 or (i + 1) == frames.shape[0]: 
            features_array = np.array(features_list)
            segment_stats = create_segment_statistics(features_array)
            array_list.append(np.hstack((features_array, np.tile(segment_stats, (seg_counter, 1)))))
            seg_counter = 0
            features_list = []


    return np.vstack(array_list)


def create_segment_statistics(segment_frames: np.ndarray) -> np.ndarray:
    # TODO: move to features.py
    def lster(energies: np.ndarray) -> float:
        thr = np.mean(energies) / 3
        return np.sum(energies < thr) / len(energies)
        
    diffs = np.abs(np.diff(segment_frames, axis=0))
    stats_list = [
        np.mean(segment_frames, axis=0),
        np.std(segment_frames, axis=0),
        np.mean(diffs, axis=0),
        np.std(diffs, axis=0),
        skew(segment_frames[:, 1]),
        skew(diffs[:, 1]),
        lster(segment_frames[:, 0])
    ]
    
    return np.hstack(stats_list)


def evaluate(clf: tree.DecisionTreeClassifier):
    ref_dict = load_reference(train=False)
    ref = np.array([literal_eval(entry['ref']) for entry in ref_dict]).reshape(-1)
    prediction = np.array([])
    for entry in ref_dict:
        file_path = f"{DATASET_PATH}/{entry['file']}"
        s = load_and_resample(file_path)
        features = create_features(s)
        pred = clf.predict(features)
        prediction = np.hstack((prediction, pred)) if prediction.size else pred

    print(f"Evaluation complete.")
    print(f"Accuracy: {accuracy_score(ref, prediction):.4f}")
    print(f"Precision: {precision_score(ref, prediction, average='macro'):.4f}")
    print(f"Recall: {recall_score(ref, prediction, average='macro'):.4f}")
    print(f"F1 Score: {f1_score(ref, prediction, average='macro'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(ref, prediction))

if __name__ == "__main__":
    main()
