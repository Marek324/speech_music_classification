# evaluator.py
# Marek Hric

import logging
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PerClassMetrics:
    f1: float
    precision: float
    recall: float


@dataclass
class SubclassMetrics:
    f1: float
    precision: float
    recall: float
    accuracy: float
    conf_mat: np.ndarray


@dataclass
class EvalResults:
    n_classes: int
    time_per_frame_ns: float
    accuracy: float
    conf_mat: np.ndarray
    by_subclass: Dict[str, SubclassMetrics]
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


@dataclass
class EvalResultsBoth:
    two_class: EvalResults
    three_class: EvalResults


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_eval(res: EvalResults) -> str:
    lines = []
    title = f"── {res.n_classes}-class evaluation "
    lines.append(title + "─" * max(0, 50 - len(title)))
    lines.append(f"  Time/frame : {res.time_per_frame_ns * 1e-6:.4f} ms")
    lines.append(f"  Macro F1   : {res.f1:.4f}")
    lines.append("")

    lines.append(f"  {'':12s}  {'F1':>6}  {'P':>6}  {'R':>6}")
    for lbl in res.labels:
        name = LABEL_NAMES.get(lbl, str(lbl))
        pc = res.per_class[lbl]
        lines.append(f"  {name:<12s}  {pc.f1:>6.4f}  {pc.precision:>6.4f}  {pc.recall:>6.4f}")
    lines.append("  " + "─" * 37)
    lines.append(f"  {'Macro':<12s}  {res.f1:>6.4f}  {res.precision:>6.4f}  {res.recall:>6.4f}")
    lines.append("")

    col_labels = [LABEL_NAMES.get(l, str(l)) for l in res.labels]
    col_w = max(len(n) for n in col_labels) + 2
    row_label_w = max(len(n) for n in col_labels) + 2
    header = " " * (row_label_w + 4) + "".join(f"{n:>{col_w}}" for n in col_labels)
    lines.append("  Confusion matrix (rows=true, cols=pred):")
    lines.append("  " + header)
    for i, lbl in enumerate(res.labels):
        row_name = LABEL_NAMES.get(lbl, str(lbl))
        row_vals = "".join(f"{res.conf_mat[i, j]:>{col_w}}" for j in range(len(res.labels)))
        lines.append(f"  {row_name:<{row_label_w}}  [{row_vals}  ]")

    return "\n".join(lines)


def _fmt_subclass_table(by_subclass: Dict[str, SubclassMetrics]) -> str:
    if not by_subclass:
        return ""
    lines = ["── By subclass " + "─" * 36]
    lines.append(f"  {'Subclass':<24}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Acc':>6}")
    for sub, res in sorted(by_subclass.items()):
        lines.append(
            f"  {sub:<24}  {res.f1:>6.4f}  {res.precision:>6.4f}  {res.recall:>6.4f}  {res.accuracy:>6.4f}"
        )
    return "\n".join(lines)


def format_report(both: EvalResultsBoth) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "\n\n".join([
        f"Evaluation report — {ts}",
        _fmt_eval(both.two_class),
        _fmt_eval(both.three_class),
        _fmt_subclass_table(both.three_class.by_subclass),
    ])


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return p, r, f1


def _compute_subclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sub: np.ndarray,
    eval_labels: list,
) -> Dict[str, SubclassMetrics]:
    global_fp = {c: int(np.sum((y_pred == c) & (y_true != c))) for c in eval_labels}

    by_sub = {}
    for sub in np.unique(y_sub):
        mask = y_sub == sub
        y_t, y_p = y_true[mask], y_pred[mask]
        sub_labels = [l for l in eval_labels if l in set(y_t.tolist())]

        prec_vals, rec_vals, f1_vals = [], [], []
        for c in sub_labels:
            tp = int(np.sum((y_t == c) & (y_p == c)))
            fn = int(np.sum((y_t == c) & (y_p != c)))
            p, r, f1 = _prf(tp, global_fp[c], fn)
            prec_vals.append(p)
            rec_vals.append(r)
            f1_vals.append(f1)

        by_sub[str(sub)] = SubclassMetrics(
            f1=float(np.mean(f1_vals)) if f1_vals else 0.0,
            precision=float(np.mean(prec_vals)) if prec_vals else 0.0,
            recall=float(np.mean(rec_vals)) if rec_vals else 0.0,
            accuracy=accuracy_score(y_t, y_p),
            conf_mat=confusion_matrix(y_t, y_p, labels=sub_labels),
        )

    return by_sub


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sub: np.ndarray,
    eval_labels: list,
    time_per_sample_ns: float,
) -> EvalResults:
    f1_per = f1_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)

    per_class = {
        lbl: PerClassMetrics(f1=float(f1_per[i]), precision=float(prec_per[i]), recall=float(rec_per[i]))
        for i, lbl in enumerate(eval_labels)
    }

    return EvalResults(
        n_classes=len(eval_labels),
        time_per_frame_ns=time_per_sample_ns,
        accuracy=float(accuracy_score(y_true, y_pred)),
        conf_mat=confusion_matrix(y_true, y_pred, labels=eval_labels),
        by_subclass=_compute_subclass_metrics(y_true, y_pred, y_sub, eval_labels),
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
    Labels: -1=speech, 1=music, 2=inactive.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(subclasses):
        raise ValueError("y_true, y_pred and subclasses must have the same length")

    mask_2 = np.isin(y_true, [-1, 1])
    both = EvalResultsBoth(
        two_class=_compute_metrics(y_true[mask_2], y_pred[mask_2], subclasses[mask_2], [-1, 1], time_per_sample_ns),
        three_class=_compute_metrics(y_true, y_pred, subclasses, [-1, 1, 2], time_per_sample_ns),
    )

    report = format_report(both)
    if save_to_file:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{output_name}.eval"
        out_path.write_text(report)
        log.info("Saved evaluation results to %s", out_path)

    log.info("\n%s", report)
    return both
