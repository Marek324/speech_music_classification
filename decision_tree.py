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
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

from defaults import (
    DATASET_PATH,
    FRAME_LEN_MS,
    SEGMENT_LEN_S
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
    features = np.array([])
    frames = framing(s)
    f_prev = None
    statistics = np.array([])
    seg_len = SEGMENT_LEN_S * 1000 // FRAME_LEN_MS
    seg_counter = 0

    for i, f in enumerate(frames):
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

        seg_counter += 1
        if (i + 1) % seg_len == 0 or (i + 1) == frames.shape[0]: 
            segment_stats = create_segment_statistics(features[-seg_counter:,:])
            statistics = np.vstack((statistics, np.tile(segment_stats, (seg_counter, 1)))) if statistics.size else np.tile(segment_stats, (seg_counter, 1))
            seg_counter = 0


    features = np.hstack((features, statistics))

    return features

def create_segment_statistics(segment_frames: np.ndarray) -> np.ndarray:
    stats = np.mean(segment_frames, axis=0) # means
    stats = np.hstack((stats, np.std(segment_frames, axis=0))) # std deviations
    # TODO: continue from 2.4.ii
    
    return stats


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
