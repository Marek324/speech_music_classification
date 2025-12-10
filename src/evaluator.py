# evaluator.py
# Marek Hric

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from modelclass import ModelClass


class Evaluator:
    def __init__(self):
        pass

    def eval(self, model: ModelClass, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        y_pred = model.predict_batch(X)

        acc = accuracy_score(y, y_pred)
        conf = confusion_matrix(y, y_pred)
        cls_report = classification_report(y, y_pred, output_dict=True)

        try:
            y_proba = model.predict_proba_batch(X)
            ll = log_loss(y, y_proba)
        except Exception:
            ll = None

        return {
            "accuracy": acc,
            "confusion_matrix": conf,
            "classification_report": cls_report,
            "log_loss": ll,
        }
