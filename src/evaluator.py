# evaluator.py
# Marek Hric

import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

log = logging.getLogger(__name__)


@dataclass
class SubClassEvalResults:
    f1: float
    accuracy: float
    precision: float
    recall: float
    conf_mat: np.ndarray


@dataclass
class EvalResults:
    n_classes: int
    time_per_frame_ns: float  # extract + classify per frame (ns)
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
            f"      OVERALL EVALUATION ({self.n_classes}-class)  ",
            "================================",
            f"Time per frame: {self.time_per_frame_ns * 1e-6:.4f} ms\n"
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

        report.append("\n\n")

        return "\n".join(report)


@dataclass
class EvalResultsBoth:
    """Results of 2-class and 3-class evaluation."""

    two_class: EvalResults
    three_class: EvalResults

    def __str__(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"Evaluation report — {ts}\n{'=' * 40}\n{self.two_class}\n{self.three_class}"


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sub: np.ndarray,
    eval_labels: list,
    time_per_sample_ns: float,
) -> EvalResults:
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
            precision=precision_score(y_t_s, y_p_s, average="macro", zero_division=0),
            recall=recall_score(y_t_s, y_p_s, average="macro", zero_division=0),
            conf_mat=confusion_matrix(y_t_s, y_p_s, labels=eval_labels),
        )

    return EvalResults(
        n_classes=len(eval_labels),
        time_per_frame_ns=time_per_sample_ns,
        f1=f1_overall,
        accuracy=acc_overall,
        precision=prec_overall,
        recall=rec_overall,
        conf_mat=conf_overall,
        by_subclass=by_sub,
        labels=eval_labels,
    )


def run_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subclasses: np.ndarray,
    time_per_sample_ns: float,
    output_name: str,
    save_to_file: bool = True,
) -> EvalResultsBoth:
    """
    Main evaluation entry point. Computes metrics and optionally saves report.

    Labels: -1=speech, 1=music, 2=inactive (classic format).
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(subclasses):
        raise ValueError("y_true, y_pred and subclasses must have the same length")

    # 2-class: speech vs music only
    mask_2 = np.isin(y_true, [-1, 1])
    res_2 = _compute_metrics(
        y_true[mask_2], y_pred[mask_2], subclasses[mask_2], [-1, 1], time_per_sample_ns
    )

    # 3-class: speech, music, inactive
    res_3 = _compute_metrics(
        y_true, y_pred, subclasses, [-1, 1, 2], time_per_sample_ns
    )

    both = EvalResultsBoth(two_class=res_2, three_class=res_3)

    if save_to_file:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{output_name}.eval"
        with open(out_path, "w") as f:
            f.write(str(both))
        log.info("Saved evaluation results to %s", out_path)

    log.info("\n%s", both)
    return both


class Evaluator:
    """Adapter for models with predict_batch(X). Used by classic smoke-test."""

    def eval(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        subclasses: np.ndarray,
        save_to_file: bool = True,
        extract_time_per_frame_ns: float = 0.0,
    ) -> EvalResultsBoth:
        """Run evaluation via model.predict_batch(X)."""
        if X.shape[0] != y.shape[0] or X.shape[0] != subclasses.shape[0]:
            raise ValueError("X, y and subclasses must have the same size")

        start = time.perf_counter_ns()
        y_pred = model.predict_batch(X)
        classify_ns = time.perf_counter_ns() - start
        time_per_sample_ns = (classify_ns / len(X)) + extract_time_per_frame_ns

        return run_evaluation(
            y,
            y_pred,
            subclasses,
            time_per_sample_ns,
            output_name=model.name,
            save_to_file=save_to_file,
        )
