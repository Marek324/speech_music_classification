# evaluator.py
# Marek Hric

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .common import subclass_primary

log = logging.getLogger(__name__)

LABEL_NAMES = {-1: "Speech", 1: "Music", 2: "Inactive"}

from .seed import RAND_SEED as BOOTSTRAP_SEED

BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PerClassMetrics:
    f1: float
    precision: float
    recall: float
    f1_ci: tuple[float, float] | None = None  # (lower, upper) at 95%


@dataclass
class SubclassMetrics:
    f1: float
    precision: float
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
    weighted_f1: float = 0.0
    macro_auroc: float | None = None
    macro_f1_ci: tuple[float, float] | None = None  # (lower, upper) at 95%
    device: str = "cpu"
    y_scores: np.ndarray | None = None

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

def _fmt_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return ""
    lo, hi = ci
    return f"  [{lo:.4f}, {hi:.4f}]"


def _fmt_eval(res: EvalResults) -> str:
    lines = []
    title = "── Evaluation "
    lines.append(title + "─" * max(0, 50 - len(title)))
    lines.append(f"  Device     : {res.device}")
    lines.append(f"  ms/frame   : {res.time_per_frame_ns * 1e-6:.4f}")
    lines.append(f"  Macro F1   : {res.f1:.4f}{_fmt_ci(res.macro_f1_ci)}")
    lines.append(f"  Weighted F1: {res.weighted_f1:.4f}")
    lines.append(f"  Accuracy   : {res.accuracy:.4f}")
    if res.macro_auroc is not None:
        lines.append(f"  Macro AUROC: {res.macro_auroc:.4f}")
    lines.append("")

    lines.append(f"  {'Class':<12s}  {'F1':>6}  {'P':>6}  {'R':>6}  {'F1 CI':>18}")
    for lbl in res.labels:
        name = LABEL_NAMES.get(lbl, str(lbl))
        pc = res.per_class[lbl]
        ci_str = f"[{pc.f1_ci[0]:.4f}, {pc.f1_ci[1]:.4f}]" if pc.f1_ci else ""
        lines.append(f"  {name:<12s}  {pc.f1:>6.4f}  {pc.precision:>6.4f}  {pc.recall:>6.4f}  {ci_str:>18}")
    lines.append("  " + "─" * 57)
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
    lines.append(f"  {'Subclass':<24}  {'F1':>6}  {'P':>6}  {'R':>6}")
    for sub, res in sorted(by_subclass.items()):
        lines.append(f"  {sub:<24}  {res.f1:>6.4f}  {res.precision:>6.4f}  {res.recall:>6.4f}")
    lines.append("  (F1/P/R of each subclass's primary class within its frames)")
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
    """Per-subclass F1/P/R of the subclass's primary class, within its frame mask.

    `primary` is determined by `subclass_primary(name)` — uniform denominator across
    subclasses (one label, binary-on-mask), so values are directly comparable.
    """
    by_sub = {}
    for sub in np.unique(y_sub):
        sub_name = str(sub)
        mask = y_sub == sub
        y_t, y_p = y_true[mask], y_pred[mask]

        if len(y_t) == 0:
            by_sub[sub_name] = SubclassMetrics(f1=0.0, precision=0.0, recall=0.0)
            continue

        try:
            primary = subclass_primary(sub_name)
        except ValueError:
            log.warning("Unknown subclass %r; skipping primary-class metric", sub_name)
            by_sub[sub_name] = SubclassMetrics(f1=0.0, precision=0.0, recall=0.0)
            continue

        y_t_bin = (y_t == primary).astype(np.int8)
        y_p_bin = (y_p == primary).astype(np.int8)

        f1 = float(f1_score(y_t_bin, y_p_bin, zero_division=0))
        prec = float(precision_score(y_t_bin, y_p_bin, zero_division=0))
        rec = float(recall_score(y_t_bin, y_p_bin, zero_division=0))

        by_sub[sub_name] = SubclassMetrics(f1=f1, precision=prec, recall=rec)

    return by_sub


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clip_ids: np.ndarray,
    eval_labels: list,
    n_draws: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    ci: float = BOOTSTRAP_CI,
) -> tuple[tuple[float, float], Dict[int, tuple[float, float]]]:
    """Clip-level bootstrap CI on macro-F1 and per-class F1.

    Resamples unique clip IDs with replacement (n_draws times), concatenates all frames
    of the drawn clips, recomputes F1 per draw. Returns (macro_ci, {label: per_class_ci}).
    """
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(clip_ids)
    n_clips = len(unique_ids)

    # Index frames by clip for fast resampling.
    sort_idx = np.argsort(clip_ids, kind="stable")
    sorted_ids = clip_ids[sort_idx]
    sorted_true = y_true[sort_idx]
    sorted_pred = y_pred[sort_idx]
    # Start/end of each clip in the sorted view.
    starts = np.searchsorted(sorted_ids, unique_ids, side="left")
    ends = np.searchsorted(sorted_ids, unique_ids, side="right")

    macro_draws = np.empty(n_draws, dtype=np.float64)
    per_class_draws = np.empty((n_draws, len(eval_labels)), dtype=np.float64)

    for d in range(n_draws):
        draw = rng.integers(0, n_clips, size=n_clips)
        # Gather all frames from sampled clips (variable-length — flatten via concat).
        parts_true = [sorted_true[starts[i]:ends[i]] for i in draw]
        parts_pred = [sorted_pred[starts[i]:ends[i]] for i in draw]
        yt = np.concatenate(parts_true)
        yp = np.concatenate(parts_pred)
        per = f1_score(yt, yp, labels=eval_labels, average=None, zero_division=0)
        per_class_draws[d] = per
        macro_draws[d] = float(np.mean(per))

    alpha = (1.0 - ci) / 2.0
    lo_q, hi_q = 100.0 * alpha, 100.0 * (1.0 - alpha)

    macro_ci = (float(np.percentile(macro_draws, lo_q)), float(np.percentile(macro_draws, hi_q)))
    per_class_ci: Dict[int, tuple[float, float]] = {}
    for i, lbl in enumerate(eval_labels):
        lo = float(np.percentile(per_class_draws[:, i], lo_q))
        hi = float(np.percentile(per_class_draws[:, i], hi_q))
        per_class_ci[lbl] = (lo, hi)

    return macro_ci, per_class_ci


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_labels: list,
    time_per_frame_ns: float,
    by_subclass: Dict[str, SubclassMetrics],
    device: str = "cpu",
    y_scores: np.ndarray | None = None,
    macro_f1_ci: tuple[float, float] | None = None,
    per_class_f1_ci: Dict[int, tuple[float, float]] | None = None,
) -> EvalResults:
    f1_per = f1_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)

    per_class = {
        lbl: PerClassMetrics(
            f1=float(f1_per[i]),
            precision=float(prec_per[i]),
            recall=float(rec_per[i]),
            f1_ci=(per_class_f1_ci or {}).get(lbl),
        )
        for i, lbl in enumerate(eval_labels)
    }

    weighted = float(f1_score(y_true, y_pred, labels=eval_labels, average="weighted", zero_division=0))

    macro_auroc: float | None = None
    if y_scores is not None and len(y_scores) == len(y_true):
        # Per-class OVR AUROC, averaged. Computed manually because TCN outputs
        # independent sigmoid scores (per paper §3.2) that don't sum to 1, which
        # sklearn's multi_class="ovr" requires.
        per_class_auc = []
        for i, lbl in enumerate(eval_labels):
            y_bin = (y_true == lbl).astype(np.int8)
            if y_bin.any() and not y_bin.all():
                try:
                    per_class_auc.append(float(roc_auc_score(y_bin, y_scores[:, i])))
                except ValueError as e:
                    log.warning("AUROC for label %s unavailable: %s", lbl, e)
        if len(per_class_auc) == len(eval_labels):
            macro_auroc = float(np.mean(per_class_auc))

    return EvalResults(
        n_classes=len(eval_labels),
        time_per_frame_ns=time_per_frame_ns,
        accuracy=float(accuracy_score(y_true, y_pred)),
        conf_mat=confusion_matrix(y_true, y_pred, labels=eval_labels),
        by_subclass=by_subclass,
        labels=eval_labels,
        per_class=per_class,
        weighted_f1=weighted,
        macro_auroc=macro_auroc,
        macro_f1_ci=macro_f1_ci,
        device=device,
    )


def run_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subclasses: np.ndarray,
    time_per_frame_ns: float,
    output_name: str,
    save_to_file: bool = True,
    device: str = "cpu",
    output_dir: Path | None = None,
    y_scores: np.ndarray | None = None,
    clip_ids: np.ndarray | None = None,
) -> EvalResults:
    """
    Main evaluation entry point. Computes metrics and optionally saves report.
    Labels: -1=speech, 1=music, 2=inactive.

    *output_dir* overrides the default ``repo_root/results/`` save location.
    Pass it to route experiment results into experiment-specific subdirectories.

    *y_scores*: optional (N, 3) array of continuous scores (columns: speech,
    music, inactive). If provided and *save_to_file* is True, saved as
    ``<output_name>_scores.npz`` alongside the ``.eval`` report. Also used
    to compute macro AUROC.

    *clip_ids*: optional per-frame int array identifying the source clip of each
    frame. When provided, a 1000-draw clip-level bootstrap produces 95% CIs on
    macro-F1 and per-class F1.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(subclasses):
        raise ValueError("y_true, y_pred and subclasses must have the same length")
    if clip_ids is not None and len(clip_ids) != len(y_true):
        raise ValueError("clip_ids must match y_true length")

    eval_labels = [-1, 1, 2]
    by_subclass = _compute_subclass_metrics(y_true, y_pred, subclasses)

    macro_ci: tuple[float, float] | None = None
    per_class_ci: Dict[int, tuple[float, float]] | None = None
    if clip_ids is not None and len(clip_ids) > 0:
        macro_ci, per_class_ci = _bootstrap_ci(
            y_true, y_pred, clip_ids, eval_labels,
        )

    res = _compute_metrics(
        y_true, y_pred, eval_labels, time_per_frame_ns, by_subclass,
        device=device, y_scores=y_scores,
        macro_f1_ci=macro_ci, per_class_f1_ci=per_class_ci,
    )
    res.y_scores = y_scores

    report = format_report(res)
    if save_to_file:
        results_dir = output_dir if output_dir is not None else (
            Path(__file__).resolve().parent.parent / "results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{output_name}.eval"
        out_path.write_text(report)
        log.info("Saved evaluation results to %s", out_path)

        if y_scores is not None:
            scores_path = results_dir / f"{output_name}_scores.npz"
            np.savez(scores_path, y_true=y_true, y_scores=y_scores)
            log.info("Saved scores to %s", scores_path)

    log.info("\n%s", report)
    return res
