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

LABEL_NAMES = {-1: "Speech", 1: "Music", 2: "Inactive"}


@dataclass
class PerClassMetrics:
    f1: float
    precision: float
    recall: float


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
    accuracy: float
    conf_mat: np.ndarray
    by_subclass: Dict[str, SubClassEvalResults]
    labels: list
    per_class: Dict[int, PerClassMetrics]

    @property
    def f1(self) -> float:
        return float(np.mean([self.per_class[l].f1 for l in self.labels]))

    @property
    def precision(self) -> float:
        return float(np.mean([self.per_class[l].precision for l in self.labels]))

    @property
    def recall(self) -> float:
        return float(np.mean([self.per_class[l].recall for l in self.labels]))

    def __str__(self):
        lines = []
        title = f"── {self.n_classes}-class evaluation "
        lines.append(title + "─" * max(0, 50 - len(title)))
        lines.append(f"  Time/frame : {self.time_per_frame_ns * 1e-6:.4f} ms")
        lines.append(f"  Accuracy   : {self.accuracy:.4f}")
        lines.append("")

        # Per-class table
        lines.append(f"  {'':12s}  {'F1':>6}  {'P':>6}  {'R':>6}")
        for lbl in self.labels:
            name = LABEL_NAMES.get(lbl, str(lbl))
            pc = self.per_class[lbl]
            lines.append(f"  {name:<12s}  {pc.f1:>6.4f}  {pc.precision:>6.4f}  {pc.recall:>6.4f}")
        lines.append("  " + "─" * 37)
        lines.append(f"  {'Macro':<12s}  {self.f1:>6.4f}  {self.precision:>6.4f}  {self.recall:>6.4f}")
        lines.append("")

        # Labelled confusion matrix
        col_labels = [LABEL_NAMES.get(l, str(l)) for l in self.labels]
        col_w = max(len(n) for n in col_labels) + 2
        row_label_w = max(len(n) for n in col_labels) + 2

        header = " " * (row_label_w + 4) + "".join(f"{n:>{col_w}}" for n in col_labels)
        lines.append("  Confusion matrix (rows=true, cols=pred):")
        lines.append("  " + header)
        for i, lbl in enumerate(self.labels):
            row_name = LABEL_NAMES.get(lbl, str(lbl))
            row_vals = "".join(f"{self.conf_mat[i, j]:>{col_w}}" for j in range(len(self.labels)))
            lines.append(f"  {row_name:<{row_label_w}}  [{row_vals}  ]")

        return "\n".join(lines)


@dataclass
class EvalResultsBoth:
    """Results of 2-class and 3-class evaluation."""

    two_class: EvalResults
    three_class: EvalResults

    def __str__(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        header = f"Evaluation report — {ts}"
        subclass_lines = _format_subclass_table(self.three_class.by_subclass)
        return f"{header}\n\n{self.two_class}\n\n{self.three_class}\n\n{subclass_lines}"


def _format_subclass_table(by_subclass: Dict[str, SubClassEvalResults]) -> str:
    if not by_subclass:
        return ""
    lines = ["── By subclass " + "─" * 36]
    header = f"  {'Subclass':<24}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Acc':>6}"
    lines.append(header)
    for sub, res in sorted(by_subclass.items()):
        lines.append(
            f"  {sub:<24}  {res.f1:>6.4f}  {res.precision:>6.4f}  {res.recall:>6.4f}  {res.accuracy:>6.4f}"
        )
    return "\n".join(lines)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sub: np.ndarray,
    eval_labels: list,
    time_per_sample_ns: float,
    accuracy_mask: np.ndarray | None = None,
) -> EvalResults:
    f1_per = f1_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)

    per_class = {
        lbl: PerClassMetrics(f1=float(f1_per[i]), precision=float(prec_per[i]), recall=float(rec_per[i]))
        for i, lbl in enumerate(eval_labels)
    }

    if accuracy_mask is not None:
        acc_overall = accuracy_score(y_true[accuracy_mask], y_pred[accuracy_mask])
    else:
        acc_overall = accuracy_score(y_true, y_pred)

    conf_overall = confusion_matrix(y_true, y_pred, labels=eval_labels)

    by_sub = {}
    for sub in np.unique(y_sub):
        s_mask = y_sub == sub
        y_t_s, y_p_s = y_true[s_mask], y_pred[s_mask]
        sub_labels = [l for l in eval_labels if l in set(y_t_s.tolist())]
        by_sub[str(sub)] = SubClassEvalResults(
            f1=f1_score(y_t_s, y_p_s, labels=sub_labels, average="macro", zero_division=0),
            accuracy=accuracy_score(y_t_s, y_p_s),
            precision=precision_score(y_t_s, y_p_s, labels=sub_labels, average="macro", zero_division=0),
            recall=recall_score(y_t_s, y_p_s, labels=sub_labels, average="macro", zero_division=0),
            conf_mat=confusion_matrix(y_t_s, y_p_s, labels=sub_labels),
        )

    return EvalResults(
        n_classes=len(eval_labels),
        time_per_frame_ns=time_per_sample_ns,
        accuracy=acc_overall,
        conf_mat=conf_overall,
        by_subclass=by_sub,
        labels=eval_labels,
        per_class=per_class,
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

    # Single pass over all frames with all 3 labels
    full = _compute_metrics(y_true, y_pred, subclasses, [-1, 1, 2], time_per_sample_ns)

    # 2-class: reuse per-class values from full; accuracy filtered to speech/music frames only
    mask_2 = np.isin(y_true, [-1, 1])
    conf_2 = full.conf_mat[np.ix_([0, 1], [0, 1])]
    res_2 = EvalResults(
        n_classes=2,
        labels=[-1, 1],
        per_class={l: full.per_class[l] for l in [-1, 1]},
        conf_mat=conf_2,
        accuracy=accuracy_score(y_true[mask_2], y_pred[mask_2]),
        by_subclass=full.by_subclass,
        time_per_frame_ns=full.time_per_frame_ns,
    )
    res_3 = full

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
