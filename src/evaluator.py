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
    def __init__(self, class_labels: Optional[List[int]] = None):
        self.class_labels = class_labels if class_labels is not None else [-1, 1, 2]

    def eval(
        self, model: ModelClass, X: np.ndarray, y: np.ndarray, subclasses: np.ndarray
    ) -> EvalResults:
        y_pred = model.predict_batch(X)

        f1_overall = f1_score(y, y_pred, average="macro", zero_division=0)
        conf_overall = confusion_matrix(y, y_pred, labels=self.class_labels)
        by_sub = {}

        for sub in np.unique(subclasses):
            mask = subclasses == sub

            y_true_sub = y[mask]
            y_pred_sub = y_pred[mask]

            by_sub[str(sub)] = SubClassEvalResults(
                f1=f1_score(y_true_sub, y_pred_sub, average="macro", zero_division=0),
                conf_mat=confusion_matrix(
                    y_true_sub, y_pred_sub, labels=self.class_labels
                ),
            )

        return EvalResults(f1=f1_overall, conf_mat=conf_overall, by_subclass=by_sub)
