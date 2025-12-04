# dataset.py
# Marek Hric

import numpy as np


class SMDataset:
    """
    Holds extracted features
    """

    def __init__(self, X_speech: np.ndarray, X_music: np.ndarray):
        self.speech: np.ndarray = X_speech
        self.music: np.ndarray = X_music
        assert self.speech.shape[1] == self.music.shape[1]
        self.n_feat = self.speech.shape[1]
        self.xs: np.ndarray = np.vstack((X_speech, X_music))
        self.targets: np.ndarray = np.hstack(
            (np.ones(len(X_speech)).astype(int) * -1, np.ones(len(X_music)).astype(int))
        )
