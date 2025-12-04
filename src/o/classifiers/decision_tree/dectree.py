# decision_tree.py
# Marek Hric


import numpy as np
from scipy.stats import gaussian_kde

from classifiers import SMClassifier
from sm_lib import SMDataset

from .threshold_computation import feature_thresholds
from .feature_selection import select_features
from .threshold_comparison import comp_thresholds


class SMDecisionTree(SMClassifier):
    def __init__(self):
        # {feat_i: {'sx' : {'thr': float, 'metrics': {'I': float, 'Er': float}}}}
        self.thresholds: dict[
            int, dict[str, dict[str, float | bool | dict[str, float]]]
        ] = {}
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

    def predict(self, x: np.ndarray) -> float:
        assert x.ndim == 1 and x.size > 0  # x = sample
        # duplicate code from frame_labeling.py:60
        E = np.sum(x**2)
        if E < 0.2:  # silence
            return 2

        comp: dict[str, int] = comp_thresholds(x, self.top_features, self.thresholds)

        ALPHA = 0.80

        if (
            (comp["sx"] > 0 and comp["mx"] == comp["mh"] == 0)
            or (comp["sx"] > 1 and comp["mx"] == 0)
            or (comp["sh"] > ALPHA * self.n_top_feat and comp["mh"] == 0)
        ):
            return -1
        elif (
            (comp["mx"] > 0 and comp["sx"] == comp["sh"] == 0)
            or (comp["mx"] > 1 and comp["sx"] == 0)
            or (comp["mh"] > ALPHA * self.n_top_feat and comp["sh"] == 0)
        ):
            return 1
        else:
            return (comp["ms"] - comp["ss"]) / self.n_top_feat
