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

    def fit(self, X: np.ndarray, y: np.ndarray):
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
        modeldumppath = f"{os.path.dirname(__file__)}/../model_dump"
        os.makedirs(f"{modeldumppath}", exist_ok=True)
        fullpath = f"{modeldumppath}/{self.name}"
        joblib.dump(self.tree, fullpath)

    def load(self):
        modeldumppath = f"{os.path.dirname(__file__)}/../model_dump"
        fullpath = f"{modeldumppath}/{self.name}"
        os.path.exists(fullpath)
        self.tree = joblib.load(fullpath)
