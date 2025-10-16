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
        self.x: np.ndarray = np.vstack((X_speech, X_music))
        self.targets: np.ndarray = np.vstack((
            np.ones(len(X_speech)).astype(int),
            np.ones(len(X_music)).astype(int) + 1
        ))
