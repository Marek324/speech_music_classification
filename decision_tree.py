# decision_tree.py
# Marek Hric

import sys

try:
    import numpy as np
    from scipy.stats import gaussian_kde
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)


class DecisionTree:
    def __init__(self):
        self.n_features = None
        self.speech_pdfs = {} # {feat_i: pdf}
        self.music_pdfs = {}
        self.thresholds = {} # {feat_i: {'ex_speech' : {'val': float, 'metrics': {'I': float, 'Er': float}}}}

    def train(self, X_speech: np.ndarray, X_music: np.ndarray):
        print(X_speech.shape)
        print(X_music.shape)

        self.n_features = X_speech.shape[1]
        for i in range(self.n_features):
            S = X_speech[:, i]
            M = X_music[:, i]

            self.speech_pdfs[i] = gaussian_kde(S)
            self.music_pdfs[i] = gaussian_kde(M)

            self._comp_thresholds(X_speech, X_music)

    def _comp_thresholds(self, X_speech: np.ndarray, X_music: np.ndarray):
        pass

    def _comp_ex_thr(self, target: np.ndarray, opp: np.ndarray, dir: str):
        if dir == 'min':
            thr = np.min(opp)
            valid = target[target < thr]
            return np.max(valid) if len(valid) > 0 else thr
        else:
            thr = np.max(opp)
            valid = target[target > thr]
            return np.min(valid) if len(valid) > 0 else thr


        








