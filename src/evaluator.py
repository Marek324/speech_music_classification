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
    # y_true is clip-uniform within a subclass (sourced from the dataset `class`
    # column), so fp is structurally 0 inside the mask. Precision collapses to
    # 1.0/0.0, accuracy equals recall, and F1 = 2R/(1+R) — only recall carries
    # independent information.
    recall: float


def _device_label(is_cuda: bool) -> str:
    if is_cuda:
        try:
            import torch
            return torch.cuda.get_device_name()
        except Exception:
            return "cuda"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


@dataclass
class EvalResults:
    n_classes: int
    time_per_frame_ns: float
    accuracy: float
    conf_mat: np.ndarray
    by_subclass: Dict[str, SubclassMetrics]
    labels: list
    per_class: Dict[int, PerClassMetrics]
    device: str = "cpu"

    @property
    def f1(self) -> float:
        return float(np.mean([self.per_class[l].f1 for l in self.labels]))

    @property
    def precision(self) -> float:
        return float(np.mean([self.per_class[l].precision for l in self.labels]))

    @property
    def recall(self) -> float:
        return float(np.mean([self.per_class[l].recall for l in self.labels]))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_eval(res: EvalResults) -> str:
    lines = []
    title = "── Evaluation "
    lines.append(title + "─" * max(0, 50 - len(title)))
    lines.append(f"  Device     : {res.device}")
    lines.append(f"  ms/frame   : {res.time_per_frame_ns * 1e-6:.4f}")
    lines.append(f"  Macro F1   : {res.f1:.4f}")
    lines.append("")

    lines.append(f"  {'Class':<12s}  {'F1':>6}  {'P':>6}  {'R':>6}")
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
    lines.append(f"  {'Subclass':<24}  {'R':>6}")
    for sub, res in sorted(by_subclass.items()):
        lines.append(f"  {sub:<24}  {res.recall:>6.4f}")
    return "\n".join(lines)


def format_report(res: EvalResults) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "\n\n".join([
        f"Evaluation report — {ts}",
        _fmt_eval(res),
        _fmt_subclass_table(res.by_subclass),
    ])


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _compute_subclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sub: np.ndarray,
) -> Dict[str, SubclassMetrics]:
    by_sub = {}
    for sub in np.unique(y_sub):
        mask = y_sub == sub
        y_t, y_p = y_true[mask], y_pred[mask]

        # y_true is clip-uniform within a subclass — primary is any frame's label.
        primary = int(y_t[0])
        recall = float(np.mean(y_p == primary)) if len(y_t) else 0.0

        by_sub[str(sub)] = SubclassMetrics(recall=recall)

    return by_sub


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_labels: list,
    time_per_sample_ns: float,
    by_subclass: Dict[str, SubclassMetrics],
    device: str = "cpu",
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
        by_subclass=by_subclass,
        labels=eval_labels,
        per_class=per_class,
        device=device,
    )


def run_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subclasses: np.ndarray,
    time_per_sample_ns: float,
    output_name: str,
    save_to_file: bool = True,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> EvalResults:
    """
    Main evaluation entry point. Computes metrics and optionally saves report.
    Labels: -1=speech, 1=music, 2=inactive.

    *output_dir* overrides the default ``repo_root/results/`` save location.
    Pass it to route experiment results into experiment-specific subdirectories.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(subclasses):
        raise ValueError("y_true, y_pred and subclasses must have the same length")

    by_subclass = _compute_subclass_metrics(y_true, y_pred, subclasses)
    res = _compute_metrics(y_true, y_pred, [-1, 1, 2], time_per_sample_ns, by_subclass, device=device)

    report = format_report(res)
    if save_to_file:
        results_dir = output_dir if output_dir is not None else (
            Path(__file__).resolve().parent.parent / "results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{output_name}.eval"
        out_path.write_text(report)
        log.info("Saved evaluation results to %s", out_path)

    log.info("\n%s", report)
    return res
