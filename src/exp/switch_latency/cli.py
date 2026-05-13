# src/exp/switch_latency/cli.py
# Marek Hric

"""Switch-latency benchmark on the crit-set switching clips.

For every (model × clip), the experiment:

1. Resets the runner so the streaming buffers + smoothing windows start empty.
2. Pushes ``silence_prefill_ms`` of zeros at the model's native sample rate
   so feature rings + smoothing deques are saturated; predictions during
   this window are discarded.
3. Streams the clip in one ``runner.push()`` call (which internally iterates
   per hop), collecting one Prediction per emitted hop.
4. Builds a per-hop ground-truth array from the manifest's frame-ms labels
   and finds every transition (``gt[t] != gt[t-1]``).
5. For each transition, records the smallest ``k ≥ 0`` such that the streamed
   prediction matches the new GT class for ``consecutive_match`` hops in a
   row; the latency is ``k * hop_ms``. Search halts at the next GT change;
   if no flip happens before then the event is a *miss* — excluded from the
   median, counted separately.

Per (model × cadence × variant × direction) the report stores:
  - median + p90 latency over flipped transitions (ms)
  - miss% — fraction of GT transitions the model never followed in time
  - drift% — fraction where the prediction was already wrong just before
             the transition (pre-flip-class != old GT class). High drift
             means latencies are biased low because the model wasn't on
             the right answer to begin with.

Outputs:
  - ``results/switch_latency.md`` — methodology + cadence/direction tables
  - ``results/switch_latency.svg`` — heatmap + scaling curves

CPU is pinned to 1 intra-op thread (``torch.set_num_threads(1)`` +
``OMP_NUM_THREADS=1`` env) so the NN family is on the same footing as the
sklearn classics, which don't benefit from threading at batch=1. Mirrors
the complexity benchmark's single-thread convention.

Run with ``uv run smclassifier exp switch-latency run``.
"""
from __future__ import annotations

import json
import logging
import os
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Pin BLAS threads BEFORE numpy / torch are imported anywhere downstream.
# Some BLAS backends only honour these env vars at import time.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import click
import librosa
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_CFG_PATH = _HERE / "config.toml"
_RESULTS_DIR = _HERE / "results"
_REPO_ROOT = _HERE.parents[2]  # src/exp/switch_latency → src/exp → src → repo

# Display label encoding used by ``src/demo/runner.py``:
#   -1 = speech, 0 = background, +1 = music.
_LABEL_OF = {"speech": -1, "music": 1, "background": 0}
_LABEL_NAME = {-1: "speech", 1: "music", 0: "background"}

_MODEL_SHORT = {
    "tcn": "TCN", "tcn_l": "TCN-L", "tcn_s": "TCN-S",
    "decision_tree": "DT", "gmm": "GMM", "svm": "SVM",
}
_MODEL_COLOR = {
    "tcn": "#2196F3", "tcn_l": "#E91E63", "tcn_s": "#4DB6AC",
    "decision_tree": "#4CAF50", "gmm": "#9C27B0", "svm": "#FF9800",
}


# ---------------------------------------------------------------------------
# Config + clip discovery
# ---------------------------------------------------------------------------


def _load_cfg() -> dict:
    """Read the ``[switch_latency]`` block from ``config.toml``."""
    with open(_CFG_PATH, "rb") as f:
        return tomllib.load(f)["switch_latency"]


@dataclass
class ClipMeta:
    key: str
    file: Path
    cadence_ms: int
    variant_type: str       # "2class" or "3class"
    variant_idx: int
    labels: list[dict]      # [{"label": "speech", "start": ms, "end": ms}, ...]


def _parse_switching_subclass(sub: str) -> tuple[str, int, int] | None:
    """Decode ``switching_500ms_v3`` / ``switching_3class_500ms_v3`` → (variant_type, cadence_ms, variant_idx)."""
    if not sub.startswith("switching_"):
        return None
    if sub.startswith("switching_3class_"):
        variant_type = "3class"
        tail = sub[len("switching_3class_"):]
    else:
        variant_type = "2class"
        tail = sub[len("switching_"):]
    cad_str, _, vstr = tail.partition("_v")
    if not cad_str.endswith("ms") or not vstr:
        return None
    try:
        return variant_type, int(cad_str[:-2]), int(vstr)
    except ValueError:
        return None


def _load_clip_metadata(manifest_dir: Path, cadence_set: set[int]) -> list[ClipMeta]:
    """Walk the manifest and return ClipMeta for every switching entry at an enabled cadence."""
    manifest_path = manifest_dir / "manifest.toml"
    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)
    out: list[ClipMeta] = []
    for key, fields in data.items():
        if not isinstance(fields, dict):
            continue
        parsed = _parse_switching_subclass(str(fields.get("subclass", "")))
        if parsed is None:
            continue
        variant_type, cad_ms, v_idx = parsed
        if cad_ms not in cadence_set:
            continue
        wav_path = (manifest_dir / fields["file"]).resolve()
        out.append(ClipMeta(
            key=key,
            file=wav_path,
            cadence_ms=cad_ms,
            variant_type=variant_type,
            variant_idx=v_idx,
            labels=list(fields.get("labels", [])),
        ))
    return out


# ---------------------------------------------------------------------------
# Per-clip ground truth & latency
# ---------------------------------------------------------------------------


def _ground_truth_per_hop(variant_type: str, cadence_ms: int,
                          n_hops: int, hop_ms: float) -> np.ndarray:
    """Build per-hop GT from the cadence cycle, ignoring intra-turn sub-labels.

    2-class clips cycle `speech → music`; 3-class clips cycle
    `speech → music → background`. Each cadence_ms slice is assigned its
    turn's primary class — intra-speech-turn breath labels are ignored so
    only the cadence-scheduled transitions appear in the GT, which is what
    switch-latency is meant to measure.
    """
    if variant_type == "2class":
        cycle = [_LABEL_OF["speech"], _LABEL_OF["music"]]
    else:
        cycle = [_LABEL_OF["speech"], _LABEL_OF["music"], _LABEL_OF["background"]]
    cad_hops_f = cadence_ms / hop_ms
    gt = np.zeros(n_hops, dtype=np.int8)
    for k in range(int(np.ceil(n_hops / cad_hops_f))):
        s = int(round(k * cad_hops_f))
        e = min(n_hops, int(round((k + 1) * cad_hops_f)))
        if e > s:
            gt[s:e] = cycle[k % len(cycle)]
    return gt


@dataclass
class TransitionResult:
    cadence_ms: int
    variant_type: str
    direction: str          # f"{old_name}->{new_name}"
    hops_to_flip: int | None  # None if missed
    miss: bool
    drift: bool             # pred just before transition != old GT class


def _transitions(
    gt: np.ndarray,
    pred: np.ndarray,
    hop_ms: float,
    consecutive_match: int,
) -> list[TransitionResult]:
    """Find every GT change and measure hops-to-flip for each.

    Search for hop ``t`` with ``gt[t] != gt[t-1]`` and requires
    ``pred[t+k : t+k+K] == gt[t]`` for some ``k ≥ 0``. Capped at the next
    GT change.
    """
    if gt.size < 2 or pred.size < 2:
        return []
    n = min(gt.size, pred.size)
    gt = gt[:n]
    pred = pred[:n]
    change_idx = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    if change_idx.size == 0:
        return []
    seg_ends = np.concatenate((change_idx[1:], [n]))

    out: list[TransitionResult] = []
    K = consecutive_match
    for t, t_next in zip(change_idx, seg_ends):
        old_lbl = int(gt[t - 1])
        new_lbl = int(gt[t])
        direction = f"{_LABEL_NAME[old_lbl]}->{_LABEL_NAME[new_lbl]}"
        drift = int(pred[t - 1]) != old_lbl

        # Need K consecutive matches starting at some t+k, all within
        # [t, t_next). k can range from 0 to (t_next - t - K).
        max_k_plus_1 = (t_next - t) - (K - 1)
        flipped_k: int | None = None
        if max_k_plus_1 > 0:
            window_starts = np.arange(t, t + max_k_plus_1)
            # For each candidate start s, check pred[s : s+K] all equal new_lbl.
            # Vectorised with stride tricks would be faster but K is tiny.
            for s in window_starts:
                if np.all(pred[s : s + K] == new_lbl):
                    flipped_k = int(s - t)
                    break
        out.append(TransitionResult(
            cadence_ms=0,  # filled by caller (clip-level info)
            variant_type="",
            direction=direction,
            hops_to_flip=flipped_k,
            miss=flipped_k is None,
            drift=drift,
        ))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class Bucket:
    latencies_ms: list[float] = field(default_factory=list)
    n_total: int = 0
    n_miss: int = 0
    n_drift: int = 0


def _bucket_key(cadence_ms: int, direction: str | None) -> tuple:
    """2-class + 3-class transitions are pooled — variant type is not part of the key."""
    return (cadence_ms, direction)


def _aggregate(
    results_by_model: dict[str, list[tuple[ClipMeta, list[TransitionResult]]]],
    hop_ms_by_model: dict[str, float],
) -> dict[str, dict[tuple, Bucket]]:
    """Pivot raw per-transition results into ``{model: {(cad, dir): Bucket}}``.

    Two granularities are stored per (model, cadence) — 2-class and 3-class
    transitions are pooled:
      - direction-broken-out: ``dir`` field is the literal ``"old->new"``.
      - direction-pooled: ``dir`` field is ``None``.
    """
    out: dict[str, dict[tuple, Bucket]] = {m: {} for m in results_by_model}
    for model, clips_results in results_by_model.items():
        hop_ms = hop_ms_by_model[model]
        for clip, transitions in clips_results:
            for tr in transitions:
                cad = clip.cadence_ms
                for direction in (tr.direction, None):
                    key = _bucket_key(cad, direction)
                    bucket = out[model].setdefault(key, Bucket())
                    bucket.n_total += 1
                    if tr.miss:
                        bucket.n_miss += 1
                    else:
                        bucket.latencies_ms.append(tr.hops_to_flip * hop_ms)
                    if tr.drift:
                        bucket.n_drift += 1
    return out


# ---------------------------------------------------------------------------
# Streaming driver
# ---------------------------------------------------------------------------


def _stream_clip(runner, clip: ClipMeta, silence_prefill_ms: float) -> np.ndarray:
    """Reset the runner, push silence prefill, then stream the clip.

    Returns the per-hop ``y_pred`` array (length ≈ clip_samples / chunk_samples).
    """
    sr = runner.sr
    chunk_samples = runner.chunk_samples
    audio, _ = librosa.load(str(clip.file), sr=sr, mono=True)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    runner.reset()

    prefill_samples = int(silence_prefill_ms / 1000.0 * sr)
    if prefill_samples > 0:
        # Push silence in one shot — push() iterates per hop internally,
        # so this is equivalent to feeding a long stream of zeros.
        runner.push(np.zeros(prefill_samples, dtype=np.float32))

    # Count how many prefill hops we just collected; reset internal frame
    # counter to 0 so post-prefill predictions index from the clip's start.
    # We discard the prefill predictions by simply not reading them — the
    # next push starts fresh predictions, and we use that list as y_pred.
    # (The runner's _frames counter keeps incrementing but we don't use it
    # for indexing.)
    preds = runner.push(audio)
    return np.array([p.label for p in preds], dtype=np.int8)


def _run_one_model(
    model_name: str,
    clips: list[ClipMeta],
    silence_prefill_ms: float,
    consecutive_match: int,
) -> tuple[list[tuple[ClipMeta, list[TransitionResult]]], float]:
    """Load model, stream every clip, return per-clip transition results + hop_ms."""
    from ...demo.runner import RUNNERS

    log.info("[%s] loading model ...", model_name)
    t0 = time.perf_counter()
    runner = RUNNERS[model_name]()
    log.info("[%s] loaded in %.1fs; sr=%d hop=%d (%.2f ms)",
             model_name, time.perf_counter() - t0,
             runner.sr, runner.chunk_samples,
             runner.chunk_samples / runner.sr * 1000.0)

    hop_ms = runner.chunk_samples / runner.sr * 1000.0
    out: list[tuple[ClipMeta, list[TransitionResult]]] = []
    for i, clip in enumerate(clips, 1):
        try:
            t_clip = time.perf_counter()
            y_pred = _stream_clip(runner, clip, silence_prefill_ms)
            elapsed = time.perf_counter() - t_clip

            n_hops = y_pred.size
            gt = _ground_truth_per_hop(clip.variant_type, clip.cadence_ms,
                                       n_hops, hop_ms)
            transitions = _transitions(gt, y_pred, hop_ms, consecutive_match)
            # Stamp clip-level metadata for downstream aggregation.
            for tr in transitions:
                tr.cadence_ms = clip.cadence_ms
                tr.variant_type = clip.variant_type
            out.append((clip, transitions))
            if i % 8 == 0 or i == len(clips):
                log.info("[%s] %d/%d clips (last %.1fs)", model_name, i, len(clips), elapsed)
        except Exception as exc:
            log.error("[%s] clip %s failed: %s", model_name, clip.key, exc)

    runner.close()
    return out, hop_ms


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt_cell(bucket: Bucket) -> str:
    """Format a bucket as ``med (p90) [miss%, drift%]`` for a Markdown cell."""
    if bucket.n_total == 0:
        return "—"
    if not bucket.latencies_ms:
        return f"miss 100% (n={bucket.n_total})"
    arr = np.asarray(bucket.latencies_ms)
    med = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    miss_pct = 100.0 * bucket.n_miss / bucket.n_total
    drift_pct = 100.0 * bucket.n_drift / bucket.n_total
    cell = f"{med:.0f} (p90 {p90:.0f})"
    extras = []
    if miss_pct >= 0.5:
        extras.append(f"miss {miss_pct:.0f}%")
    if drift_pct >= 5.0:
        extras.append(f"drift {drift_pct:.0f}%")
    if extras:
        cell += f", {', '.join(extras)}"
    return cell


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(["---:"] + ["---:"] * (len(header) - 1)) + "|")
    # First column left-align — model name.
    out[1] = "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _build_report(
    cfg: dict,
    agg: dict[str, dict[tuple, Bucket]],
    n_clips_by_key: dict[tuple[str, int], int],
    hop_ms_by_model: dict[str, float],
) -> str:
    """Render the Markdown report (methodology + pooled tables).

    Pools 2-class and 3-class switching transitions into a single set per
    (model × cadence) and (model × cadence × direction).
    """
    cadences = cfg["cadences_ms"]
    models = list(agg.keys())

    total_clips_per_cad = {
        cad: sum(n for (vt, c), n in n_clips_by_key.items() if c == cad)
        for cad in cadences
    }

    lines: list[str] = [
        "# Switch latency — switching crit clips",
        "",
        "## Methodology",
        "",
        "Each model streams every switching clip (2-class and 3-class pooled)",
        "via its `src/demo/runner.py` wrapper, after a per-clip reset and a",
        "silence prefill so feature rings and smoothing buffers are saturated",
        "before measurement. **Single-thread CPU** (`torch.set_num_threads(1)`",
        "+ `OMP_NUM_THREADS=1`) so every model sees the same compute budget.",
        "",
        f"- Silence prefill: **{cfg['silence_prefill_ms']:.0f} ms** of zeros at",
        "  the model's native sample rate.",
        f"- Consecutive-match window: **K = {cfg['consecutive_match']}** hops.",
        "  A transition is flipped at the smallest `k ≥ 0` such that",
        "  `y_pred[t+k : t+k+K]` all equal the new GT class. Search halts at",
        "  the next GT change.",
        "- **Latency** (ms) = `k × hop_ms` using each model's native hop",
        "  (DT 10 ms, GMM/SVM 15 ms, NN family 23.22 ms).",
        "- **Miss** (%) — fraction of GT transitions that never flipped before",
        "  the next GT change. Excluded from the median.",
        "- **Drift** (%) — fraction where the streamed prediction at the hop",
        "  just before the transition was not the *old* GT class. High drift",
        "  means the model wasn't on the right answer to begin with, and the",
        "  latency number is biased low for those events.",
        "",
        "Cells read `med (p90) [miss%, drift%]`. Miss% omitted under 0.5%,",
        "drift% omitted under 5%.",
        "",
        "## Pooled across directions (2-class + 3-class)",
        "",
        f"_Clips per cadence: {', '.join(f'{c}ms × {total_clips_per_cad[c]}' for c in cadences)}._",
        "",
    ]

    # Pooled-direction table.
    header = ["model"] + [f"{c} ms" for c in cadences]
    rows: list[list[str]] = []
    for m in models:
        cells = [_MODEL_SHORT.get(m, m)]
        for cad in cadences:
            bucket = agg[m].get((cad, None), Bucket())
            cells.append(_fmt_cell(bucket))
        rows.append(cells)
    lines.append(_md_table(header, rows))
    lines.append("")

    # Per-direction view (pooled across variants).
    directions = sorted({
        key[1] for m in models for key in agg[m] if key[1] is not None
    })
    if directions:
        lines.append("## Per direction (median ms)")
        lines.append("")
        lines.append("2-class clips contribute to `speech↔music` only; "
                     "3-class clips contribute to all six directions.")
        lines.append("")
        header2 = ["model", "cadence"] + [d.replace("->", " → ") for d in directions]
        rows2: list[list[str]] = []
        for m in models:
            for cad in cadences:
                cells = [_MODEL_SHORT.get(m, m), f"{cad} ms"]
                for direction in directions:
                    bucket = agg[m].get((cad, direction), Bucket())
                    if bucket.n_total == 0 or not bucket.latencies_ms:
                        cells.append("—")
                        continue
                    med = float(np.median(bucket.latencies_ms))
                    cells.append(f"{med:.0f}")
                rows2.append(cells)
        lines.append(_md_table(header2, rows2))
        lines.append("")

    return "\n".join(lines) + "\n"


# A cell is "untrustworthy" when miss% exceeds this — the model couldn't follow
# a meaningful fraction of transitions, so the median over the flipped subset
# isn't representative. Drift is reported as an annotation but doesn't gate
# trustworthiness on its own (it's a contamination signal, not a sufficient
# condition for rejecting the latency number).
_UNTRUSTWORTHY_MISS_PCT = 20.0


def _render_figure(
    cfg: dict,
    agg: dict[str, dict[tuple, Bucket]],
    out_path: Path,
) -> None:
    """Render 2-panel SVG: pooled latency heatmap + line plot (median vs cadence).

    Heatmap convention mirrors `scripts/visualize_results.py`: ``RdYlGn`` for
    higher-is-better metrics, ``_r`` reversed here because lower latency is
    better. Cells where ``miss% > 20`` or ``drift% > 30`` are grayed out — the
    median latency on those cells is not trustworthy (see _build_report).
    """
    cadences = cfg["cadences_ms"]
    models = list(agg.keys())

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    n_m = len(models)
    n_c = len(cadences)
    med_grid = np.full((n_m, n_c), np.nan)
    miss_grid = np.zeros((n_m, n_c))
    drift_grid = np.zeros((n_m, n_c))
    for i, m in enumerate(models):
        for j, cad in enumerate(cadences):
            bucket = agg[m].get((cad, None), Bucket())
            if bucket.n_total > 0:
                miss_grid[i, j] = 100.0 * bucket.n_miss / bucket.n_total
                drift_grid[i, j] = 100.0 * bucket.n_drift / bucket.n_total
                if bucket.latencies_ms:
                    med_grid[i, j] = float(np.median(bucket.latencies_ms))

    untrustworthy = miss_grid > _UNTRUSTWORTHY_MISS_PCT

    # Two-pass render: trustworthy cells get the RdYlGn_r colormap; untrustworthy
    # cells are overlaid in flat gray so they don't anchor the color scale.
    masked_trust = np.ma.array(med_grid, mask=untrustworthy | np.isnan(med_grid))
    masked_gray = np.ma.array(med_grid, mask=~untrustworthy | np.isnan(med_grid))

    trust_vals = med_grid[~untrustworthy & ~np.isnan(med_grid)]
    vmax = float(np.max(trust_vals)) if trust_vals.size else 1.0

    im = ax_h.imshow(masked_trust, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax)
    ax_h.imshow(masked_gray, aspect="auto",
                cmap=plt.matplotlib.colors.ListedColormap(["#bdbdbd"]),
                vmin=0, vmax=1)
    ax_h.set_xticks(range(n_c))
    ax_h.set_xticklabels([f"{c} ms" for c in cadences], fontsize=9)
    ax_h.set_yticks(range(n_m))
    ax_h.set_yticklabels([_MODEL_SHORT.get(m, m) for m in models], fontsize=9)
    ax_h.set_title("Median latency (ms) — pooled across directions and variants",
                   fontsize=10)
    for i in range(n_m):
        for j in range(n_c):
            v = med_grid[i, j]
            miss = miss_grid[i, j]
            drift = drift_grid[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.0f}"
            extra = []
            if miss >= 5:
                extra.append(f"miss {miss:.0f}%")
            if drift >= 5:
                extra.append(f"drift {drift:.0f}%")
            if extra:
                txt += "\n" + ", ".join(extra)
            if untrustworthy[i, j]:
                color = "black"
            else:
                color = "black" if (np.isnan(v) or v < vmax * 0.6) else "white"
            ax_h.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                      color=color)
    plt.colorbar(im, ax=ax_h, shrink=0.85, label="ms (lower better)")

    # Line plot: median latency vs cadence; untrustworthy points are open
    # markers so the trusted ones carry the visual weight.
    for m in models:
        ys_trust = []
        ys_open = []
        for j, cad in enumerate(cadences):
            i = models.index(m)
            v = med_grid[i, j]
            if np.isnan(v):
                ys_trust.append(np.nan)
                ys_open.append(np.nan)
                continue
            if untrustworthy[i, j]:
                ys_trust.append(np.nan)
                ys_open.append(v)
            else:
                ys_trust.append(v)
                ys_open.append(np.nan)
        color = _MODEL_COLOR.get(m, "#888")
        ax_l.plot(cadences, ys_trust, marker="o",
                  color=color, label=_MODEL_SHORT.get(m, m), lw=1.6)
        # Open markers for untrustworthy cells, dashed line connecting.
        ax_l.plot(cadences, ys_open, marker="o",
                  color=color, linestyle=":", lw=1.0,
                  markerfacecolor="white", markeredgecolor=color)
    ax_l.set_xscale("log")
    ax_l.set_xticks(cadences)
    ax_l.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax_l.set_xlabel("cadence (ms)")
    ax_l.set_ylabel("median latency (ms)")
    ax_l.set_title("Latency vs cadence (open markers = miss > 20%)")
    ax_l.grid(alpha=0.3)
    ax_l.legend(fontsize=8)

    fig.suptitle("Switch latency — silence-prefilled streaming on switching clips",
                 fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("switch-latency")
def switch_latency_group():
    """Switch-latency benchmark on the crit-set switching clips."""


def _pin_single_thread() -> None:
    """Set torch + sklearn-relevant BLAS to a single thread so every model is on
    equal footing (sklearn at batch=1 ignores threading; TCN gets a free
    speedup from torch's intra-op pool without this pin)."""
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


@switch_latency_group.command("run")
@click.option("--model", "-m", "models", multiple=True,
              help="Subset of models to bench (default: every model in config.toml).")
def run_cmd(models: tuple[str, ...]):
    """Bench every (or selected) model on every switching clip; write .md + .svg."""
    _pin_single_thread()
    cfg = _load_cfg()
    selected_models = list(models) if models else list(cfg["models"])
    manifest_dir = (_REPO_ROOT / cfg["manifest_dir"]).resolve()
    cadence_set = set(cfg["cadences_ms"])
    silence_prefill_ms = float(cfg["silence_prefill_ms"])
    consecutive_match = int(cfg["consecutive_match"])

    clips = _load_clip_metadata(manifest_dir, cadence_set)
    if not clips:
        log.error("No switching clips matched cadences %s under %s", cadence_set, manifest_dir)
        return
    cnts = Counter((c.variant_type, c.cadence_ms) for c in clips)
    log.info("Loaded %d switching clips: %s", len(clips),
             ", ".join(f"{vt}×{cad}ms={n}" for (vt, cad), n in sorted(cnts.items())))

    results_by_model: dict[str, list[tuple[ClipMeta, list[TransitionResult]]]] = {}
    hop_ms_by_model: dict[str, float] = {}
    for m in selected_models:
        log.info("\n=== %s ===", m)
        per_clip, hop_ms = _run_one_model(m, clips, silence_prefill_ms, consecutive_match)
        results_by_model[m] = per_clip
        hop_ms_by_model[m] = hop_ms

    agg = _aggregate(results_by_model, hop_ms_by_model)
    n_clips_by_key = {(vt, cad): n for (vt, cad), n in cnts.items()}

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_state(agg, n_clips_by_key, hop_ms_by_model, _RESULTS_DIR / "switch_latency.state.json")
    md = _build_report(cfg, agg, n_clips_by_key, hop_ms_by_model)
    out_md = _RESULTS_DIR / "switch_latency.md"
    out_md.write_text(md)
    log.info("saved → %s", out_md)
    click.echo(md)

    out_svg = _RESULTS_DIR / "switch_latency.svg"
    _render_figure(cfg, agg, out_svg)
    log.info("saved → %s", out_svg)


# ---------------------------------------------------------------------------
# State persistence + visualize-only command
# ---------------------------------------------------------------------------


def _save_state(
    agg: dict[str, dict[tuple, Bucket]],
    n_clips_by_key: dict[tuple[str, int], int],
    hop_ms_by_model: dict[str, float],
    out_path: Path,
) -> None:
    """Write aggregated buckets as JSON so the figure/report can be re-rendered
    without re-streaming. Direction keys are JSON-encoded as strings (or
    ``"__pooled__"`` for the direction-pooled bucket)."""
    state: dict = {
        "hop_ms_by_model": hop_ms_by_model,
        "n_clips_by_key": [{"variant": vt, "cadence_ms": cad, "n": n}
                           for (vt, cad), n in n_clips_by_key.items()],
        "by_model": {},
    }
    for m, by_key in agg.items():
        flat = []
        for (cad, direction), bucket in by_key.items():
            flat.append({
                "cadence_ms": cad,
                "direction": direction if direction is not None else "__pooled__",
                "latencies_ms": list(bucket.latencies_ms),
                "n_total": bucket.n_total,
                "n_miss": bucket.n_miss,
                "n_drift": bucket.n_drift,
            })
        state["by_model"][m] = flat
    out_path.write_text(json.dumps(state))
    log.info("saved → %s", out_path)


def _load_state(path: Path) -> tuple[
    dict[str, dict[tuple, Bucket]],
    dict[tuple[str, int], int],
    dict[str, float],
]:
    """Inverse of ``_save_state``."""
    raw = json.loads(path.read_text())
    hop_ms_by_model = {k: float(v) for k, v in raw["hop_ms_by_model"].items()}
    n_clips_by_key = {
        (e["variant"], int(e["cadence_ms"])): int(e["n"])
        for e in raw["n_clips_by_key"]
    }
    agg: dict[str, dict[tuple, Bucket]] = {}
    for m, flat in raw["by_model"].items():
        by_key: dict[tuple, Bucket] = {}
        for entry in flat:
            d = entry["direction"]
            direction = None if d == "__pooled__" else d
            by_key[(int(entry["cadence_ms"]), direction)] = Bucket(
                latencies_ms=list(entry["latencies_ms"]),
                n_total=int(entry["n_total"]),
                n_miss=int(entry["n_miss"]),
                n_drift=int(entry["n_drift"]),
            )
        agg[m] = by_key
    return agg, n_clips_by_key, hop_ms_by_model


@switch_latency_group.command("visualize")
def visualize_cmd():
    """Re-render switch_latency.{md,svg} from the saved state JSON; no streaming."""
    cfg = _load_cfg()
    state_path = _RESULTS_DIR / "switch_latency.state.json"
    if not state_path.exists():
        log.error("Missing state file %s — run `switch-latency run` first.", state_path)
        return
    agg, n_clips_by_key, hop_ms_by_model = _load_state(state_path)

    md = _build_report(cfg, agg, n_clips_by_key, hop_ms_by_model)
    out_md = _RESULTS_DIR / "switch_latency.md"
    out_md.write_text(md)
    log.info("saved → %s", out_md)

    out_svg = _RESULTS_DIR / "switch_latency.svg"
    _render_figure(cfg, agg, out_svg)
    log.info("saved → %s", out_svg)
