# decision_tree.py
# Marek Hric


import numpy as np
from scipy.stats import gaussian_kde

from ..sm_classifier import SMClassifier
from sm_lib import SMDataset

from .threshold_computation import feature_thresholds
from .feature_selection import select_features
from .threshold_comparison import comp_thresholds


class SMDecisionTree(SMClassifier):
    def __init__(self):
        #self.speech_pdfs: dict[int, gaussian_kde] = {} # {feat_i: pdf}
        #self.music_pdfs: dict[int, gaussian_kde] = {}
        # {feat_i: {'ex_speech' : {'thr': float, 'metrics': {'I': float, 'Er': float}}}}
        self.thresholds: dict[int, dict[str, dict[str, float|dict[str, float]]]] = {} 
        self.top_features: dict[str, list[int]] = {}
        self.n_top_feat: int = 5


    def fit(self, X: SMDataset):
        print(X.speech.shape)
        print(X.music.shape)
        assert X.n_feat >= self.n_top_feat

        for i in range(X.n_feat):
            S = X.speech[:, i]
            M = X.music[:, i]

            pdf_s = gaussian_kde(S)
            pdf_m = gaussian_kde(M)

            self.thresholds[i] = feature_thresholds(i, S, M, pdf_s, pdf_m)

        self.top_features = select_features(X, self.thresholds, self.n_top_feat)


    def predict(self, X: np.ndarray) -> int:
        # duplicate code from frame_labeling.py:60
        E = np.sum(X**2)
        if E < 0.2: # silence
            return 2

        comp_thresholds()


        return 1

