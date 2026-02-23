# evaluator.py
# Marek Hric

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class SubClassEvalResults:
    f1: float
    accuracy: float
    precision: float
    recall: float
    conf_mat: np.ndarray


@dataclass
class EvalResults:
    time: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    conf_mat: np.ndarray
    by_subclass: Dict[str, SubClassEvalResults]
    labels: list

    def __str__(self):
        """Generates a readable text report of the results."""
        report = [
            "================================",
            "      OVERALL EVALUATION        ",
            "================================",
            f"Classification time: {self.time:.8f}ns\n"
            f"Accuracy:  {self.accuracy:.4f}",
            f"F1 (Macro): {self.f1:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            "\nConfusion Matrix:",
            str(self.conf_mat),
            "\n================================",
            "      BY SUBCLASS RESULTS       ",
            "================================",
        ]

        for sub, res in self.by_subclass.items():
            report.append(f"\nSubclass: {sub}")
            report.append(
                f"  F1: {res.f1:.4f} | Acc: {res.accuracy:.4f} | P: {res.precision:.4f} | R: {res.recall:.4f}"
            )

        return "\n".join(report)


class Evaluator:
    def eval(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        subclasses: np.ndarray,
        n_classes: int = 3,
        save_to_file: bool = True,
    ) -> EvalResults:
        # Input validation
        if n_classes not in [2, 3]:
            raise ValueError("n_classes must be 2 or 3")
        if X.shape[0] != y.shape[0] or X.shape[0] != subclasses.shape[0]:
            raise ValueError("X, y and subclasses must have the same size")

        start = time.perf_counter_ns()
        y_pred_full = model.predict_batch(X)
        diff = time.perf_counter_ns() - start
        avg_time = float(diff) / len(X)

        if n_classes == 2:
            mask = np.isin(y, [-1, 1])
            eval_labels = [-1, 1]
        else:
            mask = np.ones(len(y), dtype=bool)
            eval_labels = [-1, 1, 2]

        y_true = y[mask]
        y_pred = y_pred_full[mask]
        y_sub = subclasses[mask]

        # Calculate Overall Metrics
        # Note: average='macro' is used for F1/Precision/Recall in multi-class settings
        f1_overall = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc_overall = accuracy_score(y_true, y_pred)
        prec_overall = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_overall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        conf_overall = confusion_matrix(y_true, y_pred, labels=eval_labels)

        by_sub = {}
        for sub in np.unique(y_sub):
            s_mask = y_sub == sub
            y_t_s, y_p_s = y_true[s_mask], y_pred[s_mask]

            by_sub[str(sub)] = SubClassEvalResults(
                f1=f1_score(y_t_s, y_p_s, average="macro", zero_division=0),
                accuracy=accuracy_score(y_t_s, y_p_s),
                precision=precision_score(
                    y_t_s, y_p_s, average="macro", zero_division=0
                ),
                recall=recall_score(y_t_s, y_p_s, average="macro", zero_division=0),
                conf_mat=confusion_matrix(y_t_s, y_p_s, labels=eval_labels),
            )

        res = EvalResults(
            time=avg_time,
            f1=f1_overall,
            accuracy=acc_overall,
            precision=prec_overall,
            recall=rec_overall,
            conf_mat=conf_overall,
            by_subclass=by_sub,
            labels=eval_labels,
        )

        with open(f"{model.name}.eval", "a") as f:
            f.write(str(res))

        return res
