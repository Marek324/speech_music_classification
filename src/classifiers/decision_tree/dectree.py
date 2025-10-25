# decision_tree.py
# Marek Hric


import numpy as np
from scipy.stats import gaussian_kde

from ..sm_classifier import SMClassifier
from sm_lib import SMDataset
from .threshold_computation import comp_thresholds

from .threshold_computation import comp_thresholds
from .feature_selection import select_features


class SMDecisionTree(SMClassifier):
    def __init__(self):
        self.speech_pdfs: dict[int, gaussian_kde] = {} # {feat_i: pdf}
        self.music_pdfs: dict[int, gaussian_kde] = {}
        # {feat_i: {'ex_speech' : {'thr': float, 'metrics': {'I': float, 'Er': float}}}}
        self.thresholds: dict[int, dict[str, dict[str, float|dict[str, float]]]] = {} 
        self.top_features: dict[str, list[int]] = {
            'ex_speech': [], 'ex_music':  [],
            'high_prob_speech': [], 'high_prob_music': [],
            'separation': []
        }
        self.n_top_feat: int = 5


    def fit(self, X: SMDataset):
        print(X.speech.shape)
        print(X.music.shape)
        assert X.n_feat >= self.n_top_feat

        for i in range(X.n_feat):
            S = X.speech[:, i]
            M = X.music[:, i]

            self.speech_pdfs[i] = gaussian_kde(S)
            self.music_pdfs[i] = gaussian_kde(M)

            comp_thresholds(self, i, S, M)

        select_features(self, X)


    def predict(self, X: np.ndarray) -> int:
        # duplicate code from frame_labeling.py:60
        E = np.sum(X**2)
        if E < 0.2: # silence
            return 2




        return 1

