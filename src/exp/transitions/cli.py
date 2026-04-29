"""Transition analysis on the switching clips of the critical set.

Two metrics, computed per (model, cadence, switching variant):

* **latency** — for each GT label change at frame ``t`` with ``y_true[t]=B``,
  the smallest ``k >= 0`` such that ``y_pred[t+k] == B``, scaled to ms via
  the model's frame hop. Search halts at the next GT change; if the model
  never matches the new label inside that segment, the event is recorded as
  a *miss* and excluded from the median (capping misses at the segment
  length would compress slow models toward the cadence). Miss percentage
  is reported alongside.
* **flicker** — number of ``y_pred`` changes per second inside stable GT
  segments, with the first ``flicker_settle_ms`` after each GT change
  excluded so settling transients don't masquerade as flicker.

Reads the ``_scores.npz`` files produced by ``smclassifier exp critical
eval``; switching clips are identified by ``subclass`` prefix
``speech_switching_``. No re-evaluation here — this module is a pure
analysis shim.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import tomli

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_CFG_PATH = _HERE / "config.toml"
_RESULTS_DIR = _HERE / "results"
_GRAPHS_DIR = _RESULTS_DIR / "graphs"
_REPO_ROOT = _HERE.parent.parent.parent

_MODEL_SHORT = {
    "tcn": "TCN", "tcn_lstm": "TCN+LSTM", "small_tcn": "SmallTCN", "smaller_tcn": "SmallerTCN",
    "decision_tree": "DT", "gmm": "GMM", "svm": "SVM",
}
_MODEL_COLOR = {
    "tcn": "#2196F3", "tcn_lstm": "#E91E63", "small_tcn": "#009688", "smaller_tcn": "#4DB6AC",
    "decision_tree": "#4CAF50", "gmm": "#9C27B0", "svm": "#FF9800",
}


# ---------------------------------------------------------------------------
# Config + score loading
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(_CFG_PATH, "rb") as f:
        return tomli.load(f)["transitions"]


def _resolve_scores_dir(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (_REPO_ROOT / raw).resolve()


def _load_model_scores(model: str, scores_dir: Path) -> dict[str, np.ndarray] | None:
    path = scores_dir / f"{model}_scores.npz"
    if not path.exists():
        log.info("[%s] no scores file at %s — skipping", model, path)
        return None
    with np.load(path, allow_pickle=False) as d:
        if "y_pred" not in d.files:
            # Reconstruct from y_scores so older npz files still work.
            idx = d["y_scores"].argmax(axis=1)
            y_pred = np.array([-1, 1, 2], dtype=np.int64)[idx]
        else:
            y_pred = d["y_pred"].astype(np.int64)
        return {
            "y_true": d["y_true"].astype(np.int64),
            "y_pred": y_pred,
            "clip_ids": d["clip_ids"].astype(np.int64),
            "subclasses": d["subclasses"].astype(str),
        }


# ---------------------------------------------------------------------------
# Switching-clip discovery
# ---------------------------------------------------------------------------

def _switching_buckets(
    subclass_by_clip: dict[int, str], cadences_ms: list[int]
) -> dict[tuple[str, int], list[int]]:
    """Return {(variant, cadence_ms): [clip_id, ...]} for every switching clip.

    variant is "2class" for `speech_switching_<cad>ms` clips,
    "3class" for `speech_switching_3class_<cad>ms`. Clips whose cadence isn't
    in ``cadences_ms`` are skipped silently.
    """
    cadence_set = set(cadences_ms)
    buckets: dict[tuple[str, int], list[int]] = {}
    for cid, sub in subclass_by_clip.items():
        if not sub.startswith("speech_switching_"):
            continue
        tail = sub[len("speech_switching_"):]
        if tail.startswith("3class_"):
            variant = "3class"
            cad_str = tail[len("3class_"):]
        else:
            variant = "2class"
            cad_str = tail
        if not cad_str.endswith("ms"):
            continue
        try:
            cad = int(cad_str[:-2])
        except ValueError:
            continue
        if cad not in cadence_set:
            continue
        buckets.setdefault((variant, cad), []).append(cid)
    return buckets


# ---------------------------------------------------------------------------
# Per-clip metrics
# ---------------------------------------------------------------------------

def _transition_latencies(
    y_true: np.ndarray, y_pred: np.ndarray, frame_ms: float
) -> tuple[list[float], int, int]:
    """For each GT change inside the clip, return (hit-latencies in ms, n_hits, n_total).

    A hit = pred matches new GT label at least once before the next GT change.
    Capping misses at the segment length would understate slow models, so they
    are tracked separately via the (n_hits, n_total) counters.
    """
    if len(y_true) < 2:
        return [], 0, 0
    change_idx = np.flatnonzero(np.diff(y_true) != 0) + 1
    if change_idx.size == 0:
        return [], 0, 0
    seg_ends = np.concatenate((change_idx[1:], [len(y_true)]))
    latencies: list[float] = []
    hits = 0
    for s, e in zip(change_idx, seg_ends):
        new_lbl = int(y_true[s])
        match = np.flatnonzero(y_pred[s:e] == new_lbl)
        if match.size == 0:
            continue
        latencies.append(float(match[0]) * frame_ms)
        hits += 1
    return latencies, hits, int(change_idx.size)


def _stable_flicker(
    y_true: np.ndarray, y_pred: np.ndarray, frame_ms: float, settle_ms: float
) -> tuple[float, float]:
    """Return (n_pred_changes, total_seconds) summed over stable GT segments.

    The first ``settle_ms`` of each GT segment are excluded; segments shorter
    than that contribute nothing.
    """
    if len(y_true) < 2:
        return 0.0, 0.0
    change_idx = np.flatnonzero(np.diff(y_true) != 0) + 1
    seg_starts = np.concatenate(([0], change_idx))
    seg_ends = np.concatenate((change_idx, [len(y_true)]))
    settle_frames = int(np.ceil(settle_ms / frame_ms))
    n_changes = 0
    total_frames = 0
    for s, e in zip(seg_starts, seg_ends):
        s_eff = s + settle_frames
        if e - s_eff < 2:
            continue
        seg_pred = y_pred[s_eff:e]
        n_changes += int(np.sum(np.diff(seg_pred) != 0))
        total_frames += int(e - s_eff)
    return float(n_changes), total_frames * frame_ms / 1000.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _grouped_bar(ax, x_labels, models, vals_by_model, ylabel, title):
    n_models = len(models)
    width = 0.82 / max(n_models, 1)
    x = np.arange(len(x_labels))
    for i, m in enumerate(models):
        vals = np.asarray(vals_by_model[m], dtype=float)
        offsets = x + (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            offsets, np.where(np.isnan(vals), 0, vals), width=width,
            color=_MODEL_COLOR.get(m, "#888"),
            edgecolor="white", linewidth=0.4,
            label=_MODEL_SHORT.get(m, m),
        )
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 0, "·",
                        ha="center", va="bottom", fontsize=7, color="#999")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def _render_summary(
    models: list[str],
    cadences: list[int],
    lat_by_key: dict[tuple[str, int, str], list[float]],
    miss_by_key: dict[tuple[str, int, str], list[int]],
    flicker_by_key: dict[tuple[str, int, str], list[float]],
    settle_ms: float,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    cad_labels = [f"{c} ms" for c in cadences]
    for col, variant in enumerate(("2class", "3class")):
        med_by_model: dict[str, list[float]] = {m: [] for m in models}
        flick_by_model: dict[str, list[float]] = {m: [] for m in models}
        for cad in cadences:
            for m in models:
                lats = lat_by_key.get((variant, cad, m), [])
                med_by_model[m].append(float(np.median(lats)) if lats else np.nan)
                n_ch, sec = flicker_by_key.get((variant, cad, m), [0.0, 0.0])
                flick_by_model[m].append(n_ch / sec if sec > 0 else np.nan)
        title_prefix = "Speech ↔ Music" if variant == "2class" else "Speech ↔ Music ↔ Inactive"
        _grouped_bar(axes[0, col], cad_labels, models, med_by_model,
                     "median latency (ms)",
                     f"{title_prefix} — transition latency")
        _grouped_bar(axes[1, col], cad_labels, models, flick_by_model,
                     "flicker rate (Hz)",
                     f"{title_prefix} — stable-region flicker")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Transition analysis on switching recordings — "
        "median latency to new GT label and stable-region flicker "
        f"(settle {int(settle_ms)} ms)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_latency_cdf(
    models: list[str],
    cadences: list[int],
    lat_by_key: dict[tuple[str, int, str], list[float]],
    out_path: Path,
):
    """Latency CDF per cadence, pooling 2-class + 3-class transitions."""
    fig, axes = plt.subplots(1, len(cadences), figsize=(3.2 * len(cadences), 3.6),
                             sharey=True)
    if len(cadences) == 1:
        axes = [axes]
    for ax, cad in zip(axes, cadences):
        for m in models:
            pooled = (
                lat_by_key.get(("2class", cad, m), [])
                + lat_by_key.get(("3class", cad, m), [])
            )
            if not pooled:
                continue
            arr = np.sort(pooled)
            y = np.arange(1, arr.size + 1) / arr.size
            ax.plot(arr, y, color=_MODEL_COLOR.get(m, "#888"), lw=1.6,
                    label=f"{_MODEL_SHORT.get(m, m)} (n={arr.size}, med={np.median(arr):.0f})")
        ax.set_xlim(0, cad)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{cad} ms cadence", fontsize=9)
        ax.set_xlabel("latency (ms)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("fraction of transitions ≤ x", fontsize=8)
    fig.suptitle("Transition-latency CDF (2-class + 3-class pooled per cadence)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _format_md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _format_report(
    models: list[str],
    cadences: list[int],
    lat_by_key: dict[tuple[str, int, str], list[float]],
    miss_by_key: dict[tuple[str, int, str], list[int]],
    flicker_by_key: dict[tuple[str, int, str], list[float]],
    settle_ms: float,
    n_clips: dict[tuple[str, int], int],
) -> str:
    cad_list = ", ".join(f"{c} ms" for c in cadences)
    model_list = ", ".join(_MODEL_SHORT.get(m, m) for m in models)
    lines: list[str] = [
        "# Transition analysis — switching recordings",
        "",
        "## Methodology",
        "",
        "### Source",
        "",
        "Reads `_scores.npz` files written by `smclassifier exp critical eval`",
        "(`src/exp/critical/results/`). Switching clips are identified by the",
        "`subclass` column with prefix `speech_switching_`:",
        "",
        "- **2-class** (Speech ↔ Music) — `speech_switching_<cadence>ms`",
        "- **3-class** (Speech ↔ Music ↔ Inactive) — `speech_switching_3class_<cadence>ms`",
        "",
        f"Cadences analysed: {cad_list}. Models: {model_list}.",
        "",
        "Each model has its own per-frame y_true / y_pred at its own hop",
        "(NN: 23.22 ms; SVM/GMM: 15 ms; DT: 10 ms). Latency and flicker are",
        "computed in each model's frame grid and reported in ms / Hz so they",
        "are directly comparable.",
        "",
        "### Latency",
        "",
        "For each GT change at frame `t` with new label `B`, find the smallest",
        "`k ≥ 0` such that `y_pred[t + k] == B`; the latency is `k · hop_ms`.",
        "Search halts at the next GT change so a delayed match cannot leak",
        "into the next stable segment. If the model never matches `B` before",
        "the next change, the event is recorded as a **miss** and excluded",
        "from the median — capping miss latencies at the segment length",
        "would compress slow models toward the cadence and hide their real",
        "cost. The miss column reports the fraction of GT changes that the",
        "model failed to follow within one segment.",
        "",
        "### Flicker",
        "",
        f"Inside each stable GT segment, the first **{int(settle_ms)} ms**",
        "are excluded as a settling margin (latency-related transients are",
        "not flicker). In the remainder, every `y_pred[i] != y_pred[i-1]` is",
        "counted; the rate is reported in Hz of stable audio. Segments",
        "shorter than the settling margin contribute nothing.",
        "",
        "### Outputs",
        "",
        "- `transitions.md` — this report",
        "- `graphs/transition_summary.svg` — grouped-bar median latency &",
        "  flicker, per cadence, split by 2-class vs 3-class",
        "- `graphs/latency_cdf.svg` — latency CDF per cadence (2-class and",
        "  3-class pooled)",
        "",
        "## Results",
        "",
    ]
    for variant in ("2class", "3class"):
        title = "Speech ↔ Music" if variant == "2class" else "Speech ↔ Music ↔ Inactive"
        lines.append(f"## {title}")
        lines.append("")
        clip_sizes = ", ".join(
            f"{c}ms × {n_clips.get((variant, c), 0)}" for c in cadences
        )
        lines.append(f"_Clips per cadence: {clip_sizes}._")
        lines.append("")

        # Latency table.
        header = ["model"] + [f"{c} ms" for c in cadences]
        rows: list[list[str]] = []
        for m in models:
            cells = [_MODEL_SHORT.get(m, m)]
            for cad in cadences:
                lats = lat_by_key.get((variant, cad, m), [])
                hits, total = miss_by_key.get((variant, cad, m), [0, 0])
                if not total:
                    cells.append("—")
                    continue
                if not lats:
                    cells.append(f"miss {100:.0f}%")
                    continue
                med = np.median(lats)
                p90 = np.percentile(lats, 90)
                miss_pct = 100 * (1 - hits / total)
                cell = f"{med:.0f} (p90 {p90:.0f})"
                if miss_pct > 0.5:
                    cell += f", miss {miss_pct:.0f}%"
                cells.append(cell)
            rows.append(cells)
        lines.append("**Median latency (ms) — `med (p90), miss%`**")
        lines.append("")
        lines.append(_format_md_table(header, rows))
        lines.append("")

        # Flicker table.
        rows = []
        for m in models:
            cells = [_MODEL_SHORT.get(m, m)]
            for cad in cadences:
                n_ch, sec = flicker_by_key.get((variant, cad, m), [0.0, 0.0])
                cells.append(f"{n_ch / sec:.2f}" if sec > 0 else "—")
            rows.append(cells)
        lines.append("**Stable-region flicker (Hz)**")
        lines.append("")
        lines.append(_format_md_table(header, rows))
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group("transitions")
def transitions_group():
    """Transition analysis: latency to new GT label + stable-region flicker on switching clips."""


@transitions_group.command("run")
def run_cmd():
    """Compute latency + flicker and write report.md / graphs/*.svg."""
    cfg = _load_cfg()
    scores_dir = _resolve_scores_dir(cfg["scores_dir"])
    cadences = list(cfg["metrics"]["cadences_ms"])
    settle_ms = float(cfg["metrics"]["flicker_settle_ms"])
    frame_ms = cfg["frame_ms"]

    log.info("Loading scores from %s", scores_dir)
    model_data: dict[str, dict[str, np.ndarray]] = {}
    for m in cfg["models"]:
        d = _load_model_scores(m, scores_dir)
        if d is not None:
            model_data[m] = d
    if not model_data:
        log.error("No model scores loaded — run `smclassifier exp critical eval` first.")
        return
    models = [m for m in cfg["models"] if m in model_data]

    # Map clip_id → subclass once (any model has it; they agree by construction).
    first_model = next(iter(model_data.values()))
    subclass_by_clip: dict[int, str] = {}
    for cid in np.unique(first_model["clip_ids"]):
        mask = first_model["clip_ids"] == cid
        subclass_by_clip[int(cid)] = str(first_model["subclasses"][mask][0])

    buckets = _switching_buckets(subclass_by_clip, cadences)
    if not buckets:
        log.error("No switching clips found in the scores files — was the crit set rebuilt without them?")
        return

    n_clips: dict[tuple[str, int], int] = {k: len(v) for k, v in buckets.items()}
    log.info(
        "Switching clips per (variant, cadence): %s",
        ", ".join(f"{v}×{c}ms={n}" for (v, c), n in sorted(n_clips.items())),
    )

    lat_by_key: dict[tuple[str, int, str], list[float]] = {}
    miss_by_key: dict[tuple[str, int, str], list[int]] = {}
    flicker_by_key: dict[tuple[str, int, str], list[float]] = {}

    for (variant, cad), clip_ids in sorted(buckets.items()):
        for cid in clip_ids:
            for m in models:
                d = model_data[m]
                clip_mask = d["clip_ids"] == cid
                y_true = d["y_true"][clip_mask]
                y_pred = d["y_pred"][clip_mask]
                fm = float(frame_ms[m])
                lats, hits, total = _transition_latencies(y_true, y_pred, fm)
                key = (variant, cad, m)
                lat_by_key.setdefault(key, []).extend(lats)
                miss = miss_by_key.setdefault(key, [0, 0])
                miss[0] += hits
                miss[1] += total
                n_ch, sec = _stable_flicker(y_true, y_pred, fm, settle_ms)
                fl = flicker_by_key.setdefault(key, [0.0, 0.0])
                fl[0] += n_ch
                fl[1] += sec

    summary_path = _GRAPHS_DIR / "transition_summary.svg"
    cdf_path = _GRAPHS_DIR / "latency_cdf.svg"
    _render_summary(models, cadences, lat_by_key, miss_by_key, flicker_by_key,
                    settle_ms, summary_path)
    log.info("saved → %s", summary_path)
    _render_latency_cdf(models, cadences, lat_by_key, cdf_path)
    log.info("saved → %s", cdf_path)

    md = _format_report(models, cadences, lat_by_key, miss_by_key, flicker_by_key,
                        settle_ms, n_clips)
    out_md = _RESULTS_DIR / "transitions.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    log.info("saved → %s", out_md)
    click.echo(md)
