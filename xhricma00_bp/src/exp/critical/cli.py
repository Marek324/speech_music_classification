# exp/critical/cli.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""Critical-set inference experiment.

Runs every configured classifier against the hand-curated `crit` tier and
produces per-recording timeline visualizations comparing true vs predicted
labels across all models. No aggregate metrics are computed — the crit tier
is positioned as a manual-review tool, not a benchmark.

The crit tier may live either as a HuggingFace dataset config (once uploaded)
or as a local parquet directory staged by
`scripts/dataset/build.py --only-critical-set`. Both routes are supported by
the canonical loaders in `src/nn/dataset.py` and `src/input_handler.py`, so
this module is a pure config-+-CLI shim — it does not fork the inference loop.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tomli

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_CFG_PATH = _HERE / "config.toml"
_RESULTS_DIR = _HERE / "results"
_GRAPHS_DIR = _RESULTS_DIR / "graphs"
_REPO_ROOT = _HERE.parent.parent.parent

_LABEL_NAME = {-1: "speech", 1: "music", 2: "background"}
_LABEL_COLOR = {-1: "#2196F3", 1: "#DC143C", 2: "#9E9E9E"}
_MODEL_SHORT = {
    "tcn_l": "TCN-L", "tcn_s": "TCN-S", "tcn": "TCN",
    "decision_tree": "DT", "svm": "SVM", "gmm": "GMM",
}
_MODEL_COLOR = {
    "tcn_l":         "#E91E63",
    "tcn":           "#2196F3",
    "tcn_s":         "#4DB6AC",
    "decision_tree": "#4CAF50",
    "svm":           "#FF9800",
    "gmm":           "#9C27B0",
}
_CLASS_Y = {-1: -1.0, 2: 0.0, 1: 1.0}
_MODEL_ORDER = ("tcn_l", "tcn", "tcn_s", "decision_tree", "svm", "gmm")

_NN_VARIANT_NAMES = {"tcn_s", "tcn_l"}
_CLASSIC_NAMES = {"decision_tree", "gmm", "svm"}



def _load_cfg() -> dict:
    with open(_CFG_PATH, "rb") as f:
        return tomli.load(f)["critical"]


def _resolve_dataset_url(raw_url: str) -> str:
    """Resolve a config URL: absolute path → as-is, relative → vs repo root,
    HF identifier → unchanged. Existence is not asserted — the canonical
    loaders raise a clear error if the directory doesn't contain shards."""
    p = Path(raw_url).expanduser()
    if p.is_absolute():
        return str(p)
    candidate = (_REPO_ROOT / raw_url).resolve()
    if candidate.exists():
        return str(candidate)
    return raw_url


def _build_dataset_override(crit_cfg: dict) -> dict:
    return {
        "url": _resolve_dataset_url(crit_cfg["dataset"]["url"]),
        "name": crit_cfg["dataset"].get("name"),
    }



def _save_scores_npz(
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None,
    clip_ids: np.ndarray,
    subclasses: np.ndarray | None = None,
) -> None:
    """Save predictions to ``{_RESULTS_DIR}/{model}_scores.npz`` in the format
    the visualizer expects. Mirrors the npz-writing block inside ``run_evaluation``
    so ``_load_scores_npz`` keeps reading the file unchanged."""
    scores_path = _RESULTS_DIR / f"{model}_scores.npz"
    extras: dict[str, np.ndarray] = {"y_pred": y_pred, "clip_ids": clip_ids}
    if subclasses is not None:
        extras["subclasses"] = subclasses
    if y_scores is not None:
        np.savez(scores_path, y_true=y_true, y_scores=y_scores, **extras)
    else:
        np.savez(scores_path, y_true=y_true, **extras)
    log.info("Saved scores to %s", scores_path)


def _infer_classic_model(model_name: str, dataset_override: dict) -> bool:
    """Run inference for ``model_name`` on the crit tier and save scores; return True on success, False if weights are missing."""
    from ...classic.evaluation import get_predictions

    try:
        y_true, y_pred, subclasses, _t, y_scores, clip_ids = get_predictions(
            model_name, dataset_override=dataset_override,
        )
    except FileNotFoundError as e:
        log.warning("[%s] skipped: %s", model_name, e)
        return False
    _save_scores_npz(model_name, y_true, y_pred, y_scores, clip_ids, subclasses)
    return True


def _infer_tcn_canonical(dataset_override: dict) -> bool:
    """Run inference for the canonical TCN on the crit tier and save scores; return True on success."""
    from ...nn.evaluation import run_nn_inference
    from ...nn.tcn.config import (
        get_config, get_preprocess_stats_path, get_weights_path,
    )
    from ...nn.tcn.evaluation import _load_tcn_model
    from ...seed import seed_all

    cfg = get_config()
    cfg["dataset"] = dataset_override
    weights = get_weights_path()
    stats = get_preprocess_stats_path()
    if not weights.exists():
        log.warning("[tcn] skipped: weights missing at %s", weights)
        return False
    try:
        seed_all()
        model = _load_tcn_model(weights_path=weights, cfg=cfg, stats_path=stats)
        y_true, y_pred, subclasses, _t, _device, y_scores, clip_ids = run_nn_inference(
            model, cfg["dataset"], cfg,
        )
    except FileNotFoundError as e:
        log.warning("[tcn] skipped: %s", e)
        return False
    _save_scores_npz("tcn", y_true, y_pred, y_scores, clip_ids, subclasses)
    return True


def _infer_tcn_variant(name: str, dataset_override: dict) -> bool:
    """Run inference for a registered TCN variant (by name) on the crit tier and save scores; return True on success."""
    from ...nn.evaluation import run_nn_inference
    from ...nn.tcn.evaluation import _load_tcn_model
    from ...nn.variants import VARIANTS
    from ...seed import seed_all

    if name not in VARIANTS:
        log.error("[%s] unknown variant — not in src/nn/variants.toml", name)
        return False
    v = VARIANTS[name]
    cfg = v.get_config()
    cfg["dataset"] = dataset_override
    weights = v.get_weights_path()
    stats = v.get_stats_path()
    if not weights.exists():
        log.warning("[%s] skipped: weights missing at %s", name, weights)
        return False
    try:
        seed_all()
        model = _load_tcn_model(weights_path=weights, cfg=cfg, stats_path=stats)
        y_true, y_pred, subclasses, _t, _device, y_scores, clip_ids = run_nn_inference(
            model, cfg["dataset"], cfg,
        )
    except FileNotFoundError as e:
        log.warning("[%s] skipped: %s", name, e)
        return False
    _save_scores_npz(name, y_true, y_pred, y_scores, clip_ids, subclasses)
    return True


def _classify_model(name: str) -> str:
    """Return 'classic', 'tcn', 'variant', or 'unknown'."""
    if name in _CLASSIC_NAMES:
        return "classic"
    if name == "tcn":
        return "tcn"
    from ...nn.variants import VARIANTS
    if name in VARIANTS:
        return "variant"
    return "unknown"


def _infer_one(model: str, dataset_override: dict) -> bool:
    """Dispatch to the correct inference driver for ``model`` and return its success flag."""
    kind = _classify_model(model)
    if kind == "classic":
        return _infer_classic_model(model, dataset_override)
    if kind == "tcn":
        return _infer_tcn_canonical(dataset_override)
    if kind == "variant":
        return _infer_tcn_variant(model, dataset_override)
    log.error("[%s] unknown model name — neither classic nor a TCN variant", model)
    return False



def _load_clip_metadata(dataset_url: str) -> list[dict[str, Any]]:
    """Return `{source, class, subclass, duration_s, n_frames_at_16k}` for each
    test clip, in the same iteration order the canonical loaders produced.

    Iteration order matters because clip_ids in `_scores.npz` are 0-indexed
    against the dataset's row order. For the local-parquet route, that order
    is `sorted(...glob...)` (see `_local_parquet_shards` in src/nn/dataset.py
    and src/input_handler.py); HF iterates them in the order it's given.
    """
    import soundfile as sf
    from ...nn.dataset import _is_local_dataset, _local_parquet_shards

    if not _is_local_dataset(dataset_url):
        from datasets import load_dataset
        ds = load_dataset(dataset_url, name="crit", split="test")
        meta: list[dict[str, Any]] = []
        for row in ds:
            audio = row["audio"]
            with sf.SoundFile(io.BytesIO(audio["bytes"])) as f:
                duration_s = len(f) / f.samplerate
            meta.append({
                "source": row.get("source", f"row_{len(meta)}"),
                "class": row["class"],
                "subclass": row.get("subclass", row["class"]),
                "duration_s": duration_s,
            })
        return meta

    shards = _local_parquet_shards(Path(dataset_url), "test")
    meta: list[dict[str, Any]] = []
    for shard in shards:
        t = pq.read_table(shard, columns=["source", "class", "subclass", "audio"])
        rows = t.to_pylist()
        for row in rows:
            audio = row["audio"]
            with sf.SoundFile(io.BytesIO(audio["bytes"])) as f:
                duration_s = len(f) / f.samplerate
            meta.append({
                "source": row["source"],
                "class": row["class"],
                "subclass": row["subclass"],
                "duration_s": duration_s,
            })
    return meta


def _load_scores_npz(model: str) -> dict[str, np.ndarray] | None:
    """Load ``{model}_scores.npz`` and return y_true/y_pred/clip_ids; ``None`` if file or clip_ids missing."""
    path = _RESULTS_DIR / f"{model}_scores.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        out = {"y_true": d["y_true"].astype(np.int64)}
        if "y_pred" in d.files:
            out["y_pred"] = d["y_pred"].astype(np.int64)
        else:
            idx = d["y_scores"].argmax(axis=1)
            out["y_pred"] = np.array([-1, 1, 2], dtype=np.int64)[idx]
        if "clip_ids" in d.files:
            out["clip_ids"] = d["clip_ids"].astype(np.int64)
        else:
            log.warning("[%s] _scores.npz has no clip_ids — per-clip viz unavailable", model)
            return None
    return out


def _rle(y: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encode → list of (start_idx, length, value)."""
    if len(y) == 0:
        return []
    changes = np.flatnonzero(np.diff(y) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(y)]))
    return [(int(s), int(e - s), int(y[s])) for s, e in zip(starts, ends)]


def _draw_timeline_panel(
    ax: "plt.Axes",
    rows: list[tuple[str, np.ndarray, float]],
    t_start_s: float,
    t_end_s: float,
) -> None:
    """One panel: ground truth as faded class-colored background; each model as a step trace on y in {-1, 0, +1}.

    rows[0] is the ground-truth row (label="ground truth"); rows[1:] are model rows keyed by the raw model name
    (e.g., "tcn_l", "decision_tree") so the renderer can look up per-model colors via _MODEL_COLOR.
    """
    gt_label, gt_y, gt_frame_ms = rows[0]
    model_rows = rows[1:]

    f_start = max(0, int(t_start_s * 1000 / gt_frame_ms))
    f_end = min(len(gt_y), int(t_end_s * 1000 / gt_frame_ms) + 1)
    if f_end > f_start:
        for s, length, val in _rle(gt_y[f_start:f_end]):
            x0 = (f_start + s) * gt_frame_ms / 1000.0
            x1 = (f_start + s + length) * gt_frame_ms / 1000.0
            ax.axvspan(x0, x1, color=_LABEL_COLOR.get(int(val), "#000"), alpha=0.18, zorder=0)

    for model_name, y, frame_ms in model_rows:
        if y is None or len(y) == 0:
            continue
        f_s = max(0, int(t_start_s * 1000 / frame_ms))
        f_e = min(len(y), int(t_end_s * 1000 / frame_ms) + 1)
        if f_e <= f_s:
            continue
        x_vals = np.arange(f_s, f_e) * frame_ms / 1000.0
        y_vals = np.array([_CLASS_Y.get(int(v), 0.0) for v in y[f_s:f_e]])
        ax.step(
            x_vals, y_vals,
            where="post",
            color=_MODEL_COLOR.get(model_name, "#000"),
            linewidth=1.5,
            alpha=0.85,
            zorder=3,
        )

    ax.set_xlim(t_start_s, t_end_s)
    ax.set_ylim(-1.4, 1.4)
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_yticklabels(["speech", "background", "music"], fontsize=8)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def _legend_handles(model_names: "Iterable[str] | None" = None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    class_h = [
        Patch(facecolor=_LABEL_COLOR[-1], alpha=0.18, edgecolor="none", label=f"GT: {_LABEL_NAME[-1]}"),
        Patch(facecolor=_LABEL_COLOR[1],  alpha=0.18, edgecolor="none", label=f"GT: {_LABEL_NAME[1]}"),
        Patch(facecolor=_LABEL_COLOR[2],  alpha=0.18, edgecolor="none", label=f"GT: {_LABEL_NAME[2]}"),
    ]
    if model_names is None:
        names: list[str] = list(_MODEL_ORDER)
    else:
        present = set(model_names)
        names = [m for m in _MODEL_ORDER if m in present]
    model_h = [
        Line2D([], [], color=_MODEL_COLOR[m], linewidth=2.0, label=_MODEL_SHORT[m])
        for m in names
    ]
    return class_h + model_h


def _safe_filename(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return "".join(c if c in keep else "_" for c in name) or "clip"


def _render_clip(
    clip_idx: int,
    clip_meta: dict[str, Any],
    rows_full: list[tuple[str, np.ndarray, float]],
    out_dir: Path,
    panel_seconds: float,
    max_panels_per_file: int,
    inches_per_second: float,
    level_height: float,
    time_range: tuple[float, float] | None = None,
    prod_mode: bool = False,
) -> list[Path]:
    """Render the timeline for one clip, splitting across multiple PNGs if needed.

    Layout: each panel covers ``panel_seconds`` of audio and shows GT as faded class-colored
    background plus each model as a step trace on y in {-1, 0, +1}. Panel vertical size is
    ``3 * level_height`` (one inch-per-level for the three y-levels) plus padding.

    When ``time_range`` is given, a single panel covering that exact window is rendered
    instead of the panel_seconds-split layout. When ``prod_mode`` is set, the suptitle
    is dropped and the file is written as PDF for thesis inclusion.
    """
    duration_s = clip_meta["duration_s"]
    if duration_s <= 0:
        log.warning("clip %d has zero duration, skipping plot", clip_idx)
        return []

    if time_range is not None:
        t_start, t_end = time_range
        if t_start >= duration_s:
            log.warning(
                "clip %d (%s) duration %.2fs < start %.2fs; skipping",
                clip_idx, clip_meta["source"], duration_s, t_start,
            )
            return []
        t_end = min(t_end, duration_s)
        panel_ranges = [(t_start, t_end)]
    else:
        n_panels = max(1, int(np.ceil(duration_s / panel_seconds)))
        panel_ranges = [(i * panel_seconds, (i + 1) * panel_seconds) for i in range(n_panels)]

    n_panels_total = len(panel_ranges)
    safe = _safe_filename(clip_meta["source"])
    panel_span = panel_ranges[0][1] - panel_ranges[0][0]
    fig_w = max(4.0, panel_span * inches_per_second)
    panel_pad = 0.7
    top_reserve = 0.4 if prod_mode else 0.7
    ext = "pdf" if prod_mode else "png"

    written: list[Path] = []
    for file_idx, panel_start in enumerate(range(0, n_panels_total, max_panels_per_file)):
        panel_end = min(panel_start + max_panels_per_file, n_panels_total)
        panels_in_file = panel_end - panel_start
        fig_h = panels_in_file * (3 * level_height + panel_pad) + top_reserve
        fig, axes = plt.subplots(
            panels_in_file, 1,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )
        for k, (t0, t1) in enumerate(panel_ranges[panel_start:panel_end]):
            _draw_timeline_panel(axes[k, 0], rows_full, t0, t1)

        if not prod_mode:
            title = (
                f"crit clip #{clip_idx} — {clip_meta['source']}  "
                f"[{clip_meta['class']}/{clip_meta['subclass']}, {duration_s:.2f}s]"
            )
            if time_range is not None:
                title += f"  window {time_range[0]:.2f}-{panel_ranges[0][1]:.2f}s"
            if n_panels_total > max_panels_per_file:
                total_files = int(np.ceil(n_panels_total / max_panels_per_file))
                title += f"  (file {file_idx + 1}/{total_files})"
            fig.suptitle(title, fontsize=10, y=1 - 0.14 / fig_h)

        legend_y = (0.14 if prod_mode else 0.38) / fig_h
        plotted_models = [name for name, _, _ in rows_full[1:]]
        handles = _legend_handles(plotted_models)
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 1 - legend_y),
        )
        fig.tight_layout(rect=(0, 0.0, 1, 1 - top_reserve / fig_h))

        if time_range is not None:
            suffix = f"_{time_range[0]:.1f}-{panel_ranges[0][1]:.1f}s"
        elif n_panels_total > max_panels_per_file:
            suffix = f"_p{file_idx + 1}"
        else:
            suffix = ""
        out_path = out_dir / f"clip{clip_idx:03d}_{safe}{suffix}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def _build_clip_rows(
    clip_idx: int,
    per_model_data: dict[str, dict[str, np.ndarray]],
    frame_ms_by_model: dict[str, float],
) -> list[tuple[str, np.ndarray, float]]:
    """Return [(row_label, y_array, frame_ms), ...] starting with ground truth."""
    rows: list[tuple[str, np.ndarray, float]] = []
    gt_added = False
    for model, data in per_model_data.items():
        mask = data["clip_ids"] == clip_idx
        if not mask.any():
            continue
        if not gt_added:
            rows.append(("ground truth", data["y_true"][mask], frame_ms_by_model[model]))
            gt_added = True
        rows.append((model, data["y_pred"][mask], frame_ms_by_model[model]))
    return rows



@click.group("critical")
def critical_group():
    """Critical-set inference: per-recording timeline plots across every classifier."""


@critical_group.command("eval")
@click.option("--model", "-m", "models", multiple=True,
              help="Subset of models to run (default: every model in config.toml).")
def eval_cmd(models: tuple[str, ...]):
    """Run inference for every (or selected) model on the crit test split and
    save per-model ``_scores.npz`` files for the timeline visualizer. No
    aggregate metric reports are written — the crit tier is positioned as a
    manual-review tool, not a benchmark."""
    cfg = _load_cfg()
    selected = list(models) if models else cfg["models"]
    override = _build_dataset_override(cfg)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info(
        "Critical inference — dataset url=%s name=%s, %d model(s)",
        override["url"], override["name"], len(selected),
    )
    ok, skipped = [], []
    for m in selected:
        if _infer_one(m, override):
            ok.append(m)
        else:
            skipped.append(m)

    log.info("\nSummary: %d inferred, %d skipped", len(ok), len(skipped))
    if skipped:
        log.info("  skipped: %s", ", ".join(skipped))


def _parse_timerange(ctx, param, value):
    if value is None:
        return None
    try:
        start_s, end_s = value.split("-", 1)
        start_f, end_f = float(start_s), float(end_s)
    except ValueError:
        raise click.BadParameter(
            "expected start_s-end_s, e.g. 0-10 or 1.5-3.2",
        )
    if end_f <= start_f:
        raise click.BadParameter("end must be greater than start")
    return start_f, end_f


@critical_group.command("visualize")
@click.option("--model", "-m", "models", multiple=True,
              help="Subset of models to plot (default: every model in config.toml). Repeatable.")
@click.option("--time", "-t", "time_range", callback=_parse_timerange,
              help="Render a single panel covering start_s-end_s (e.g. 0-10 or 1.5-3.2).")
@click.option("--clip", "-c", "clip_indices", multiple=True, type=int,
              help="Subset of clip indices to render (default: every clip). Repeatable.")
@click.option("--output", "-o", "output_dir",
              type=click.Path(file_okay=False, path_type=Path),
              help=f"Output directory (default: {_GRAPHS_DIR.relative_to(_REPO_ROOT)}).")
@click.option("--prod", "-p", "prod_mode", is_flag=True,
              help="Production output: drop the title and write PDF (for thesis inclusion).")
def visualize_cmd(
    models: tuple[str, ...],
    time_range: tuple[float, float] | None,
    clip_indices: tuple[int, ...],
    output_dir: Path | None,
    prod_mode: bool,
):
    """Render per-recording true-vs-predicted timeline SVGs from the saved `_scores.npz` files."""
    cfg = _load_cfg()
    override = _build_dataset_override(cfg)
    frame_ms_by_model: dict[str, float] = cfg["frame_ms"]
    viz = cfg["viz"]

    log.info("Loading clip metadata from %s ...", override["url"])
    clips = _load_clip_metadata(override["url"])
    log.info("Found %d test clip(s).", len(clips))

    selected_models = list(models) if models else list(cfg["models"])
    per_model_data: dict[str, dict[str, np.ndarray]] = {}
    for model in selected_models:
        data = _load_scores_npz(model)
        if data is None:
            log.info("[%s] no scores file — run `critical eval` first", model)
            continue
        n_unique = int(np.unique(data["clip_ids"]).size)
        if n_unique != len(clips):
            log.warning(
                "[%s] clip count mismatch (npz=%d, parquet=%d); plots may misalign",
                model, n_unique, len(clips),
            )
        if model not in frame_ms_by_model:
            log.warning("[%s] no frame_ms in config; skipping", model)
            continue
        per_model_data[model] = data

    if not per_model_data:
        log.error("No model scores loaded — nothing to visualize.")
        return

    selected_clips = set(clip_indices) if clip_indices else None
    if selected_clips is not None:
        out_of_range = [c for c in selected_clips if c < 0 or c >= len(clips)]
        if out_of_range:
            log.error("clip indices out of range [0, %d): %s", len(clips), out_of_range)
            return

    out_dir = Path(output_dir) if output_dir is not None else _GRAPHS_DIR
    no_filters = (
        not models and time_range is None and not clip_indices
        and output_dir is None and not prod_mode
    )
    if no_filters and out_dir.exists():
        for stale in (*out_dir.glob("clip*.svg"), *out_dir.glob("clip*.png")):
            stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    written_total: list[Path] = []
    rendered_clips = 0
    for clip_idx, meta in enumerate(clips):
        if selected_clips is not None and clip_idx not in selected_clips:
            continue
        rows = _build_clip_rows(clip_idx, per_model_data, frame_ms_by_model)
        if len(rows) <= 1:
            log.warning("clip %d (%s): no model predictions → skipping", clip_idx, meta["source"])
            continue
        written = _render_clip(
            clip_idx, meta, rows, out_dir,
            panel_seconds=float(viz["panel_seconds"]),
            max_panels_per_file=int(viz["max_panels_per_file"]),
            inches_per_second=float(viz["inches_per_second"]),
            level_height=float(viz["level_height"]),
            time_range=time_range,
            prod_mode=prod_mode,
        )
        for p in written:
            log.info("saved → %s", p)
        written_total.extend(written)
        rendered_clips += 1

    log.info(
        "\nWrote %d plot file(s) for %d clip(s) into %s",
        len(written_total), rendered_clips, out_dir,
    )


@critical_group.command("run-all")
@click.option("--model", "-m", "models", multiple=True,
              help="Subset of models to evaluate (default: every model in config.toml).")
@click.pass_context
def run_all_cmd(ctx, models):
    """Run inference for every model and render the timeline plots in one shot."""
    ctx.invoke(eval_cmd, models=models)
    ctx.invoke(
        visualize_cmd,
        models=models, time_range=None,
        clip_indices=(), output_dir=None, prod_mode=False,
    )
