# decision_tree.py
# Marek Hric

import sys

try:
    import numpy as np
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)


class DecisionTree:
    def __init__(self):
        pass

    def train(self, X_speech: np.ndarray, X_music: np.ndarray):
        print(X_speech.shape)
        print(X_music.shape)





