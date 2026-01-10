# decisiontree.py
# Marek Hric


import os

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from modelclass import ModelClass


class DecisionTree(ModelClass):
    name: str = "unnamed"

    def __init__(self, name: str):
        self.tree = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "selector",
                    SelectKBest(
                        score_func=mutual_info_classif, k=10
                    )
                ),
                ("classifier", DecisionTreeClassifier()),
            ]
        )
        self.name = name

    def _get_weights_path(self) -> str:
        path = f"{os.path.dirname(__file__)}/../weights/{self.name}"
        return path

    def fit(self, X: np.ndarray, y: np.ndarray):
        weights_path = self._get_weights_path()
        if os.path.exists(weights_path):
            print(f"Loading DecisionTree weights from {weights_path}")
            self.tree = joblib.load(weights_path)
            return

        # select speech/music
        mask = np.isin(y, [-1, 1])

        X_train = X[mask]
        y_train = y[mask]

        print("Training DecisionTree")
        self.tree.fit(X_train, y_train)

    def predict(self, frame: np.ndarray) -> int:
        return self.tree.predict(frame)

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        return self.tree.predict_proba(frame)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict_proba(X)

    def save(self):
        path = self._get_weights_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            print(f"DecisionTree weights already exist at {path}, overwriting.")
        joblib.dump(self.tree, path)

    def load(self):
        path = self._get_weights_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"DecisionTree weights not found at {path}")

        self.tree = joblib.load(path)
