# evaluator.py
# Marek Hric

from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
)

from modelclass import ModelClass


@dataclass
class SubClassEvalResults:
    f1: float
    conf_mat: np.ndarray


@dataclass
class EvalResults:
    f1: float
    conf_mat: np.ndarray
    by_subclass: Dict[str, SubClassEvalResults]


class Evaluator:
    def __init__(self):
        # We don't hardcode labels in init anymore, we decide per eval call
        pass

    def eval(
        self, 
        model: ModelClass, 
        X: np.ndarray, 
        y: np.ndarray, 
        subclasses: np.ndarray, 
        n_classes: int = 3
    ) -> EvalResults:
        
        if n_classes not in [2, 3]:
            raise ValueError("n_classes must be 2 or 3")

        if X.shape[0] != y.shape[0] or X.shape[0] != subclasses.shape[0]:
            raise ValueError("X, y and subclasses must have the same number of samples")

        y_pred_full = model.predict_batch(X)

        if n_classes == 2:
            mask = np.isin(y, [-1, 1])
            eval_labels = [-1, 1]
            print(f"Evaluator: 2-Class Mode. Ignoring {np.sum(~mask)} inactive samples.")
        else:
            mask = np.ones(len(y), dtype=bool)
            eval_labels = [-1, 1, 2]
            print("Evaluator: 3-Class Mode. Evaluating Speech vs Music vs Inactive.")

        y_true = y[mask]
        y_pred = y_pred_full[mask]
        y_sub = subclasses[mask]

        f1_overall = f1_score(y_true, y_pred, average="macro", zero_division=0)
        conf_overall = confusion_matrix(y_true, y_pred, labels=eval_labels)
        by_sub = {}

        for sub in np.unique(y_sub):
            sub_mask = y_sub == sub
            
            if np.sum(sub_mask) == 0:
                continue

            y_true_s = y_true[sub_mask]
            y_pred_s = y_pred[sub_mask]

            by_sub[str(sub)] = SubClassEvalResults(
                f1=f1_score(y_true_s, y_pred_s, average="macro", zero_division=0),
                conf_mat=confusion_matrix(y_true_s, y_pred_s, labels=eval_labels),
            )

        return EvalResults(f1=f1_overall, conf_mat=conf_overall, by_subclass=by_sub)
