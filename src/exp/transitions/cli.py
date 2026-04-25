"""Transition analysis across all six classifiers.

Treats every change in ``y_true`` as a transition event and measures three
behaviours per model:

1. Detection latency — wall-clock time between the new label becoming valid
   and the first matching prediction. **End-of-frame convention**: the
   prediction for frame ``t+k`` is emitted at time ``(t+k+1) * hop_ms``, and
   the transition occurs at the boundary at ``t * hop_ms``, so the exact
   latency is ``(k + 1) * hop_ms`` (not ``k * hop_ms`` as a naïve
   start-of-frame reading would give).
2. False transition rate — predicted-label flips inside stable y_true
   regions (≥``stable_guard_ms`` from any true transition).
3. Near-vs-stable accuracy — accuracy gap between ±``boundary_window_ms``
   around transitions and the stable region.

When ``clip_ids`` is present in a model's ``_scores.npz`` (i.e. the file was
written by the patched ``src/evaluator.py``), the latency analysis uses
**within-clip transitions only** — concatenation boundaries between distinct
test clips are excluded. The latency search also stops at the next clip
boundary, so an unresolved event can't "leak" into the next clip and look
like it eventually resolved.

Run with ``uv run smclassifier exp transitions run`` from the project root.
Outputs land in ``src/exp/transitions/results/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import click
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import tomli

_CFG_PATH = Path(__file__).parent / "config.toml"
_RESULTS_DIR = Path(__file__).parent / "results"
_OUT_GRAPHS = _RESULTS_DIR / "graphs"
_OUT_TXT = _RESULTS_DIR / "transition_analysis.txt"

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Plot styling — duplicated from scripts/visualize_results.py to keep this
# module importable without the scripts/ sys.path shim. Only the six known
# models are styled; unknown names fall back to grey.
_COLORS = {
    "tcn_lstm": "#E91E63", "small_tcn": "#009688", "tcn": "#2196F3",
    "decision_tree": "#4CAF50", "svm": "#FF9800", "gmm": "#9C27B0",
}
_MODEL_SHORT = {
    "tcn_lstm": "TCN+LSTM", "small_tcn": "SmallTCN", "tcn": "TCN",
    "decision_tree": "DT", "svm": "SVM", "gmm": "GMM",
}

_LABEL_NAMES = {-1: "Speech", 1: "Music", 2: "Inactive"}
_LABELS = [-1, 1, 2]
_PAIRS = [(a, b) for a in _LABELS for b in _LABELS if a != b]
_PAIR_LABELS = {(a, b): f"{_LABEL_NAMES[a][0]}→{_LABEL_NAMES[b][0]}" for a, b in _PAIRS}


def _load_config() -> dict:
    with open(_CFG_PATH, "rb") as f:
        return tomli.load(f)["transitions"]


def _load_model_data(model: str, scores_dir: Path) -> dict | None:
    """Load a single model's _scores.npz; returns None if file is absent.

    Always returns y_true and y_pred. ``y_pred`` is reconstructed from
    ``y_scores`` if the npz was saved before the y_pred patch. ``clip_ids``
    is included if present, otherwise the key is absent from the dict.
    """
    path = scores_dir / f"{model}_scores.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    y_true = d["y_true"].astype(np.int64)
    y_scores = d["y_scores"].astype(np.float32)
    if "y_pred" in d.files:
        y_pred = d["y_pred"].astype(np.int64)
    else:
        idx = y_scores.argmax(axis=1)
        y_pred = np.array([-1, 1, 2], dtype=np.int64)[idx]
    out = {"y_true": y_true, "y_pred": y_pred, "y_scores": y_scores}
    if "clip_ids" in d.files:
        out["clip_ids"] = d["clip_ids"]
    return out


def _find_transitions(y_true: np.ndarray, within_clip_mask: np.ndarray | None = None) -> np.ndarray:
    """Indices t where y_true[t] != y_true[t-1] (t >= 1).

    If ``within_clip_mask`` is given, restrict to changes where the boundary
    between t-1 and t lies within a single clip. The mask must have shape
    ``(n - 1,)`` with mask[i] = True iff clip_ids[i+1] == clip_ids[i].
    """
    change = np.diff(y_true) != 0
    if within_clip_mask is not None:
        change = change & within_clip_mask
    return (np.nonzero(change)[0] + 1).astype(np.int64)


def _compute_latency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transitions: np.ndarray,
    cap_ms: float,
    hop_ms: float,
    clip_ids: np.ndarray | None = None,
) -> Dict[Tuple[int, int], dict]:
    """Per-transition-pair latency arrays in ms (end-of-frame convention).

    For each transition at frame t (y_true[t-1]=A → y_true[t]=B):
    - Smallest k >= 0 with y_pred[t+k] == B → ``latency_ms = (k + 1) * hop_ms``
    - Search depth is ``k_max = floor(cap_ms / hop_ms) - 1``, so a non-missed
      latency is at most ``(k_max + 1) * hop_ms ≤ cap_ms``.
    - With ``clip_ids`` the search also halts at the next clip boundary;
      unresolved events are recorded as missed at their effective window.

    Returns ``{pair: {"latency_ms": float64 array, "missed": bool array}}``
    over all four directional pairs.
    """
    bucket: Dict[Tuple[int, int], dict] = {p: {"latency_ms": [], "missed": []} for p in _PAIRS}
    n = len(y_true)
    k_max = max(0, int(cap_ms / hop_ms) - 1)
    for t in transitions:
        a = int(y_true[t - 1])
        b = int(y_true[t])
        if (a, b) not in bucket:
            continue
        upper = min(n, t + k_max + 1)
        if clip_ids is not None and t < n:
            tail = clip_ids[t:upper]
            diff_idx = np.nonzero(tail != clip_ids[t])[0]
            if diff_idx.size > 0:
                upper = t + int(diff_idx[0])
        window = y_pred[t:upper]
        match = np.nonzero(window == b)[0]
        if match.size:
            k = int(match[0])
            bucket[(a, b)]["latency_ms"].append((k + 1) * hop_ms)
            bucket[(a, b)]["missed"].append(False)
        else:
            window_ms = (upper - t) * hop_ms
            bucket[(a, b)]["latency_ms"].append(min(cap_ms, window_ms))
            bucket[(a, b)]["missed"].append(True)
    return {
        p: {
            "latency_ms": np.asarray(v["latency_ms"], dtype=np.float64),
            "missed": np.asarray(v["missed"], dtype=bool),
        }
        for p, v in bucket.items()
    }


def _compute_false_transition_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    guard_transitions: np.ndarray,
    guard_frames: int,
    ms_per_frame: float,
) -> Tuple[float, Dict[int, float]]:
    """Predicted-label change rate inside stable true-label regions.

    ``guard_transitions`` should include ALL true label changes (incl. clip
    boundaries when those exist) so the stable region is honest about where
    audio change is happening.
    """
    n = len(y_true)
    mask = np.ones(n, dtype=bool)
    for t in guard_transitions:
        mask[max(0, t - guard_frames):min(n, t + guard_frames)] = False
    diffs = np.diff(y_pred) != 0
    valid = mask[1:] & mask[:-1]
    per_label: Dict[int, float] = {}
    for lbl in _LABELS:
        sel = valid & (y_true[1:] == lbl)
        minutes = sel.sum() * ms_per_frame / 60000.0
        per_label[lbl] = float((diffs & sel).sum() / minutes) if minutes > 0 else 0.0
    minutes_total = valid.sum() * ms_per_frame / 60000.0
    overall = float((diffs & valid).sum() / minutes_total) if minutes_total > 0 else 0.0
    return overall, per_label


def _compute_boundary_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    guard_transitions: np.ndarray,
    window_frames: int,
    guard_frames: int,
) -> Tuple[float, float]:
    """Accuracy inside ±window vs. ≥guard from any true transition."""
    n = len(y_true)
    in_window = np.zeros(n, dtype=bool)
    stable = np.ones(n, dtype=bool)
    for t in guard_transitions:
        in_window[max(0, t - window_frames):min(n, t + window_frames)] = True
        stable[max(0, t - guard_frames):min(n, t + guard_frames)] = False
    correct = (y_true == y_pred)
    acc_near = float(correct[in_window].mean()) if in_window.any() else 0.0
    acc_stable = float(correct[stable].mean()) if stable.any() else 0.0
    return acc_near, acc_stable


# ---------------------------------------------------------------------------
# Plots — same layout as the original script, swapped to per-model mode tag
# ---------------------------------------------------------------------------

def _plot_latency_cdf(ax, per_model_latency_ms: Dict[str, np.ndarray], cap_ms: float):
    for m, arr in per_model_latency_ms.items():
        if arr.size == 0:
            continue
        sorted_arr = np.sort(arr)
        y = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, y, color=_COLORS.get(m, "#888"), lw=1.8,
                label=f"{_MODEL_SHORT.get(m, m)}  (median={np.median(sorted_arr):.0f} ms)")
    ax.set_xlim(0, cap_ms)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Detection latency (ms, end-of-frame)")
    ax.set_ylabel("Fraction of transitions with latency ≤ x")
    ax.set_title("Transition detection latency — CDF (all pairs pooled)")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="lower right")


def _plot_latency_by_pair(ax, per_model_by_pair: Dict[str, Dict[Tuple[int, int], np.ndarray]], cap_ms: float):
    models = list(per_model_by_pair.keys())
    active_pairs = [p for p in _PAIRS if any(per_model_by_pair[m][p].size > 0 for m in models)]
    n_models = len(models)
    x = np.arange(len(active_pairs))
    width = 0.8 / n_models
    for i, m in enumerate(models):
        vals = [np.median(per_model_by_pair[m][p]) if per_model_by_pair[m][p].size else 0.0
                for p in active_pairs]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.95,
               color=_COLORS.get(m, "#888"), alpha=0.9, label=_MODEL_SHORT.get(m, m))
    ax.set_xticks(x)
    ax.set_xticklabels([_PAIR_LABELS[p] for p in active_pairs])
    ax.set_ylabel("Median latency (ms, end-of-frame)")
    ax.set_title("Median detection latency by transition type")
    ax.axhline(cap_ms, color="gray", linestyle=":", lw=0.8, alpha=0.6)
    ax.text(len(active_pairs) - 0.5, cap_ms,
            f"cap = {cap_ms:.0f} ms", ha="right", va="bottom", fontsize=7, color="gray")
    ax.legend(fontsize=8, ncol=n_models, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def _plot_false_transitions(ax, per_model_rate: Dict[str, float]):
    models = list(per_model_rate.keys())
    x = np.arange(len(models))
    vals = [per_model_rate[m] for m in models]
    bars = ax.bar(x, vals, 0.5, color=[_COLORS.get(m, "#888") for m in models], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_SHORT.get(m, m) for m in models])
    ax.set_ylabel("False transitions per minute")
    ax.set_title("Spurious prediction changes in stable-label regions")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)


def _plot_boundary_accuracy(ax, acc_pairs: Dict[str, Tuple[float, float]], window_ms: float):
    models = list(acc_pairs.keys())
    x = np.arange(len(models))
    width = 0.38
    near = [acc_pairs[m][0] for m in models]
    stable = [acc_pairs[m][1] for m in models]
    b1 = ax.bar(x - width / 2, stable, width, color="#7E8FA6", label="Stable regions", alpha=0.95)
    b2 = ax.bar(x + width / 2, near, width,
                color=[_COLORS.get(m, "#888") for m in models],
                label=f"Near transition (±{window_ms:.0f} ms)", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_SHORT.get(m, m) for m in models])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: near-transition window vs. stable region")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def _plot_summary(
    ax,
    per_model_latency_ms: Dict[str, np.ndarray],
    per_model_miss: Dict[str, float],
    per_model_false: Dict[str, float],
    per_model_acc: Dict[str, Tuple[float, float]],
    per_model_mode: Dict[str, str],
):
    ax.axis("off")
    models = list(per_model_latency_ms.keys())
    headers = [
        "Model", "Mode", "Transitions", "Median lat. (ms)",
        "p90 lat. (ms)", "Missed %", "False trans./min",
        "Acc. near", "Acc. stable", "Δ acc.",
    ]
    rows = []
    for m in models:
        lat = per_model_latency_ms[m]
        med = np.median(lat) if lat.size else np.nan
        p90 = np.percentile(lat, 90) if lat.size else np.nan
        near, stable = per_model_acc[m]
        rows.append([
            _MODEL_SHORT.get(m, m), per_model_mode[m], f"{lat.size}",
            f"{med:.0f}" if np.isfinite(med) else "—",
            f"{p90:.0f}" if np.isfinite(p90) else "—",
            f"{per_model_miss[m]*100:.1f}",
            f"{per_model_false[m]:.1f}",
            f"{near:.3f}", f"{stable:.3f}",
            f"{near - stable:+.3f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)
    for i, m in enumerate(models):
        tbl[(i + 1, 0)].set_facecolor(_COLORS.get(m, "#888"))
        tbl[(i + 1, 0)].set_alpha(0.4)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#e0e0e0")
        tbl[(0, j)].set_text_props(weight="bold")
    ax.set_title("Transition analysis — summary (latency: end-of-frame convention)")


def _save_solo(plot_fn, *args, path: Path, figsize=(9, 5), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, *args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"saved → {path}")


def _format_text_report(
    per_model_latency_ms: Dict[str, np.ndarray],
    per_model_by_pair_ms: Dict[str, Dict[Tuple[int, int], np.ndarray]],
    per_model_miss: Dict[str, float],
    per_model_false: Dict[str, float],
    per_model_false_by_label: Dict[str, Dict[int, float]],
    per_model_acc: Dict[str, Tuple[float, float]],
    per_model_mode: Dict[str, str],
    per_model_transition_count: Dict[str, int],
    cfg: dict,
    metrics: dict,
) -> str:
    lines: list[str] = []
    lines.append("Transition analysis — text report")
    lines.append("=" * 52)
    modes_seen = set(per_model_mode.values())
    if len(modes_seen) > 1:
        lines.append(
            "MIXED MODE: some models analysed with clip_ids, some without. "
            "Within-clip and concatenation-boundary transitions are not "
            "directly comparable across modes — see the Mode column."
        )
    elif modes_seen == {"v1"}:
        lines.append(
            "MODE: v1 (no clip_ids). Transitions include concatenation "
            "boundaries between consecutive test clips. Re-run evaluations "
            "with the patched evaluator to get within-clip-only numbers."
        )
    else:
        lines.append(
            "MODE: v2 (within-clip). Transitions exclude concatenation "
            "boundaries; latency search halts at the next clip boundary."
        )
    lines.append(
        f"Latency convention: end-of-frame, latency_ms = (k + 1) * hop_ms. "
        f"Cap = {metrics['latency_cap_ms']} ms; stable guard = "
        f"{metrics['stable_guard_ms']} ms; boundary window = ±"
        f"{metrics['boundary_window_ms']} ms."
    )
    lines.append("")

    lines.append(f"{'Model':<10} {'Mode':<5} {'Count':>7} {'med':>6} {'p90':>6} {'p99':>6} {'miss%':>6}  (latency ms)")
    for m, arr in per_model_latency_ms.items():
        if arr.size == 0:
            lines.append(f"{_MODEL_SHORT.get(m, m):<10} {per_model_mode[m]:<5} {'—':>7}")
            continue
        lines.append(
            f"{_MODEL_SHORT.get(m, m):<10} {per_model_mode[m]:<5} {arr.size:>7} "
            f"{np.median(arr):>6.0f} {np.percentile(arr, 90):>6.0f} "
            f"{np.percentile(arr, 99):>6.0f} "
            f"{per_model_miss[m]*100:>5.1f}%"
        )
    lines.append("")

    active = [p for p in _PAIRS
              if any(per_model_by_pair_ms[m][p].size > 0 for m in per_model_by_pair_ms)]
    lines.append("Median latency per transition type (ms, — = pair not present):")
    header = f"{'Model':<10} " + " ".join(f"{_PAIR_LABELS[p]:>6}" for p in active)
    lines.append(header)
    for m in per_model_by_pair_ms:
        cells = []
        for pair in active:
            arr = per_model_by_pair_ms[m][pair]
            cells.append(f"{np.median(arr):>6.0f}" if arr.size else f"{'—':>6}")
        lines.append(f"{_MODEL_SHORT.get(m, m):<10} " + " ".join(cells))
    lines.append("")

    lines.append("False transitions per minute (in stable regions):")
    lines.append(f"{'Model':<10} {'overall':>8} {'speech':>8} {'music':>8} {'inactive':>10}")
    for m in per_model_false:
        pl = per_model_false_by_label[m]
        lines.append(
            f"{_MODEL_SHORT.get(m, m):<10} {per_model_false[m]:>8.2f} "
            f"{pl[-1]:>8.2f} {pl[1]:>8.2f} {pl[2]:>10.2f}"
        )
    lines.append("")

    lines.append(
        f"Accuracy: near-transition (±{metrics['boundary_window_ms']} ms) vs stable "
        f"(guard ±{metrics['stable_guard_ms']} ms)"
    )
    lines.append(f"{'Model':<10} {'near':>7} {'stable':>7} {'Δ':>7}")
    for m, (near, stable) in per_model_acc.items():
        lines.append(f"{_MODEL_SHORT.get(m, m):<10} {near:>7.4f} {stable:>7.4f} {near - stable:>+7.4f}")

    return "\n".join(lines)


def _bench_models(cfg: dict) -> dict:
    """Drive every configured model through the analysis. Returns aggregated results."""
    metrics = cfg["metrics"]
    cap_ms = float(metrics["latency_cap_ms"])
    stable_guard_ms = float(metrics["stable_guard_ms"])
    boundary_window_ms = float(metrics["boundary_window_ms"])

    scores_dir_raw = cfg["scores_dir"]
    scores_dir = Path(scores_dir_raw)
    if not scores_dir.is_absolute():
        scores_dir = (_REPO_ROOT / scores_dir_raw).resolve()

    per_model_latency_ms: Dict[str, np.ndarray] = {}
    per_model_by_pair_ms: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    per_model_miss: Dict[str, float] = {}
    per_model_false: Dict[str, float] = {}
    per_model_false_by_label: Dict[str, Dict[int, float]] = {}
    per_model_acc: Dict[str, Tuple[float, float]] = {}
    per_model_mode: Dict[str, str] = {}
    per_model_transition_count: Dict[str, int] = {}

    for model in cfg["models"]:
        data = _load_model_data(model, scores_dir)
        if data is None:
            click.echo(f"  skipping {model}: no scores file at {scores_dir / (model + '_scores.npz')}")
            continue
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        clip_ids = data.get("clip_ids")
        hop_ms = float(cfg["ms_per_frame"][model])

        # Guard transitions = ALL y_true changes (incl. clip boundaries) — used
        # to define stable regions and the near-transition window. The audio
        # really does change at clip boundaries, so a y_pred flip there is not
        # spurious.
        all_trans = _find_transitions(y_true)

        # Latency transitions = within-clip only, when clip_ids available.
        if clip_ids is not None:
            same_clip = clip_ids[1:] == clip_ids[:-1]
            latency_trans = _find_transitions(y_true, within_clip_mask=same_clip)
            mode = "v2"
        else:
            latency_trans = all_trans
            mode = "v1"
        per_model_mode[model] = mode
        per_model_transition_count[model] = int(latency_trans.size)

        guard_frames = int(np.ceil(stable_guard_ms / hop_ms))
        window_frames = int(np.ceil(boundary_window_ms / hop_ms))

        by_pair = _compute_latency(
            y_true, y_pred, latency_trans, cap_ms, hop_ms, clip_ids=clip_ids,
        )
        all_lat = np.concatenate(
            [v["latency_ms"] for v in by_pair.values() if v["latency_ms"].size]
        ) if any(v["latency_ms"].size for v in by_pair.values()) else np.empty(0, dtype=np.float64)
        all_missed = np.concatenate(
            [v["missed"] for v in by_pair.values() if v["missed"].size]
        ) if any(v["missed"].size for v in by_pair.values()) else np.empty(0, dtype=bool)

        per_model_latency_ms[model] = all_lat
        per_model_by_pair_ms[model] = {p: v["latency_ms"] for p, v in by_pair.items()}
        per_model_miss[model] = float(all_missed.mean()) if all_missed.size else 0.0

        overall, by_label = _compute_false_transition_rate(
            y_true, y_pred, all_trans, guard_frames, hop_ms,
        )
        per_model_false[model] = overall
        per_model_false_by_label[model] = by_label
        per_model_acc[model] = _compute_boundary_accuracy(
            y_true, y_pred, all_trans, window_frames, guard_frames,
        )

        click.echo(
            f"  {_MODEL_SHORT.get(model, model):<10} mode={mode}  "
            f"trans={latency_trans.size:>6}  "
            f"med={np.median(all_lat):>5.0f}ms  "
            f"miss={per_model_miss[model]*100:>4.1f}%  "
            f"false/min={overall:>6.2f}  "
            f"acc near={per_model_acc[model][0]:.3f}/stable={per_model_acc[model][1]:.3f}"
        )

    return {
        "per_model_latency_ms": per_model_latency_ms,
        "per_model_by_pair_ms": per_model_by_pair_ms,
        "per_model_miss": per_model_miss,
        "per_model_false": per_model_false,
        "per_model_false_by_label": per_model_false_by_label,
        "per_model_acc": per_model_acc,
        "per_model_mode": per_model_mode,
        "per_model_transition_count": per_model_transition_count,
    }


@click.group("transitions")
def transitions_group():
    """Transition analysis: detection latency, false-flip rate, and boundary accuracy across all six classifiers."""


@transitions_group.command("run")
def run():
    """Analyse transitions; write results/{transition_analysis.txt, graphs/*.svg}."""
    cfg = _load_config()
    metrics = cfg["metrics"]
    cap_ms = float(metrics["latency_cap_ms"])
    boundary_window_ms = float(metrics["boundary_window_ms"])

    click.echo("benchmarking transitions across all six models ...")
    results = _bench_models(cfg)

    if not results["per_model_latency_ms"]:
        click.echo("no model data found — aborting.")
        return

    _OUT_GRAPHS.mkdir(parents=True, exist_ok=True)
    _save_solo(_plot_latency_cdf, results["per_model_latency_ms"], cap_ms,
               path=_OUT_GRAPHS / "latency_cdf.svg", figsize=(8, 5))
    _save_solo(_plot_latency_by_pair, results["per_model_by_pair_ms"], cap_ms,
               path=_OUT_GRAPHS / "latency_by_pair.svg", figsize=(10, 5.5))
    _save_solo(_plot_false_transitions, results["per_model_false"],
               path=_OUT_GRAPHS / "false_transitions.svg", figsize=(7, 5))
    _save_solo(_plot_boundary_accuracy, results["per_model_acc"],
               path=_OUT_GRAPHS / "boundary_accuracy.svg",
               figsize=(8, 5), window_ms=boundary_window_ms)
    _save_solo(_plot_summary,
               results["per_model_latency_ms"], results["per_model_miss"],
               results["per_model_false"], results["per_model_acc"],
               results["per_model_mode"],
               path=_OUT_GRAPHS / "transition_summary.svg", figsize=(14, 3.2))

    report = _format_text_report(
        results["per_model_latency_ms"], results["per_model_by_pair_ms"],
        results["per_model_miss"], results["per_model_false"],
        results["per_model_false_by_label"], results["per_model_acc"],
        results["per_model_mode"], results["per_model_transition_count"],
        cfg, metrics,
    )
    _OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    _OUT_TXT.write_text(report)
    click.echo(f"\nsaved → {_OUT_TXT}\n")
    click.echo(report)
