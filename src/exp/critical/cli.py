"""Critical-set evaluation experiment.

Runs every configured classifier against the hand-curated `crit` tier and
produces both metric reports (`.eval` + `_scores.npz`) and per-recording
timeline visualizations comparing true vs predicted labels across all models.

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
_REPO_ROOT = _HERE.parent.parent.parent  # /home/marek/bp

# Per-label palette: speech=blue, music=crimson, inactive=grey
# (matches `src/demo/ui/ui.py::_LABEL_COLOR` semantically).
_LABEL_NAME = {-1: "speech", 1: "music", 2: "inactive"}
_LABEL_COLOR = {-1: "#2196F3", 1: "#DC143C", 2: "#9E9E9E"}
_MODEL_SHORT = {
    "tcn_lstm": "TCN+LSTM", "small_tcn": "SmallTCN", "tcn": "TCN",
    "decision_tree": "DT", "svm": "SVM", "gmm": "GMM",
}

_NN_VARIANT_NAMES = {"small_tcn", "tcn_lstm"}  # populated lazily from VARIANTS
_CLASSIC_NAMES = {"decision_tree", "gmm", "svm"}


# ---------------------------------------------------------------------------
# Config + dataset URL resolution
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eval drivers — one per backend
# ---------------------------------------------------------------------------

def _eval_classic_model(model_name: str, dataset_override: dict) -> bool:
    from ...classic.evaluation import eval_classic

    try:
        eval_classic(
            model_name=model_name,
            dataset_override=dataset_override,
            output_name=model_name,
            output_dir=_RESULTS_DIR,
        )
        return True
    except FileNotFoundError as e:
        log.warning("[%s] skipped: %s", model_name, e)
        return False


def _eval_tcn_canonical(dataset_override: dict) -> bool:
    from ...nn.tcn.config import (
        get_config, get_preprocess_stats_path, get_weights_path,
    )
    from ...nn.tcn.evaluation import eval_tcn

    cfg = get_config()
    cfg["dataset"] = dataset_override
    weights = get_weights_path()
    stats = get_preprocess_stats_path()
    if not weights.exists():
        log.warning("[tcn] skipped: weights missing at %s", weights)
        return False
    try:
        eval_tcn(
            cfg=cfg,
            weights_path=weights,
            stats_path=stats,
            output_name="tcn",
            output_dir=_RESULTS_DIR,
        )
        return True
    except FileNotFoundError as e:
        log.warning("[tcn] skipped: %s", e)
        return False


def _eval_tcn_variant(name: str, dataset_override: dict) -> bool:
    from ...nn.tcn.evaluation import eval_tcn
    from ...nn.variants import VARIANTS

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
        eval_tcn(
            cfg=cfg,
            weights_path=weights,
            stats_path=stats,
            output_name=name,
            output_dir=_RESULTS_DIR,
        )
        return True
    except FileNotFoundError as e:
        log.warning("[%s] skipped: %s", name, e)
        return False


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


def _eval_one(model: str, dataset_override: dict) -> bool:
    kind = _classify_model(model)
    if kind == "classic":
        return _eval_classic_model(model, dataset_override)
    if kind == "tcn":
        return _eval_tcn_canonical(dataset_override)
    if kind == "variant":
        return _eval_tcn_variant(model, dataset_override)
    log.error("[%s] unknown model name — neither classic nor a TCN variant", model)
    return False


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

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
        # HF route — load via the same Dataset object the loaders would use
        # so iteration order is consistent.
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
    """One panel showing ground truth + each model as a stack of label-colored bands."""
    n_rows = len(rows)
    for r_idx, (label_text, y, frame_ms) in enumerate(rows):
        if y is None or len(y) == 0:
            continue
        f_start = max(0, int(t_start_s * 1000 / frame_ms))
        f_end = min(len(y), int(t_end_s * 1000 / frame_ms) + 1)
        if f_end <= f_start:
            continue
        slice_y = y[f_start:f_end]
        y_pos = n_rows - r_idx - 1
        bars: list[tuple[float, float]] = []
        colors: list[str] = []
        for s, length, val in _rle(slice_y):
            # `s` is offset within the slice; absolute frame index = f_start + s.
            x_start = (f_start + s) * frame_ms / 1000.0
            width = length * frame_ms / 1000.0
            bars.append((x_start, width))
            colors.append(_LABEL_COLOR.get(int(val), "#000"))
        ax.broken_barh(bars, (y_pos + 0.08, 0.84), facecolors=colors, edgecolor="none")
    ax.set_xlim(t_start_s, t_end_s)
    ax.set_ylim(0, n_rows)
    ax.set_yticks([n_rows - r - 0.5 for r in range(n_rows)])
    ax.set_yticklabels([row[0] for row in rows], fontsize=8)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")


def _legend_handles():
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=_LABEL_COLOR[-1], edgecolor="none", label=_LABEL_NAME[-1]),
        Patch(facecolor=_LABEL_COLOR[1], edgecolor="none", label=_LABEL_NAME[1]),
        Patch(facecolor=_LABEL_COLOR[2], edgecolor="none", label=_LABEL_NAME[2]),
    ]


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
    row_height: float,
) -> list[Path]:
    """Render the timeline for one clip, splitting across multiple SVGs if needed."""
    duration_s = clip_meta["duration_s"]
    if duration_s <= 0:
        log.warning("clip %d has zero duration, skipping plot", clip_idx)
        return []

    n_panels_total = max(1, int(np.ceil(duration_s / panel_seconds)))
    safe = _safe_filename(clip_meta["source"])
    n_rows = len(rows_full)
    fig_w = max(4.0, panel_seconds * inches_per_second)

    written: list[Path] = []
    for file_idx, panel_start in enumerate(range(0, n_panels_total, max_panels_per_file)):
        panels_in_file = min(max_panels_per_file, n_panels_total - panel_start)
        fig_h = panels_in_file * (n_rows * row_height + 0.7) + 0.6
        fig, axes = plt.subplots(
            panels_in_file, 1,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )
        for k in range(panels_in_file):
            global_panel = panel_start + k
            t0 = global_panel * panel_seconds
            t1 = (global_panel + 1) * panel_seconds
            _draw_timeline_panel(axes[k, 0], rows_full, t0, t1)

        title = (
            f"crit clip #{clip_idx} — {clip_meta['source']}  "
            f"[{clip_meta['class']}/{clip_meta['subclass']}, {duration_s:.2f}s]"
        )
        if n_panels_total > max_panels_per_file:
            total_files = int(np.ceil(n_panels_total / max_panels_per_file))
            title += f"  (file {file_idx + 1}/{total_files})"
        fig.suptitle(title, fontsize=10)
        fig.legend(
            handles=_legend_handles(),
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))

        suffix = "" if n_panels_total <= max_panels_per_file else f"_p{file_idx + 1}"
        out_path = out_dir / f"clip{clip_idx:03d}_{safe}{suffix}.svg"
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
    # Ground truth — pick from the first model that loaded; all of them carry
    # the same y_true sequence aligned to their own clip_ids.
    gt_added = False
    for model, data in per_model_data.items():
        mask = data["clip_ids"] == clip_idx
        if not mask.any():
            continue
        if not gt_added:
            rows.append(("ground truth", data["y_true"][mask], frame_ms_by_model[model]))
            gt_added = True
        rows.append((_MODEL_SHORT.get(model, model), data["y_pred"][mask], frame_ms_by_model[model]))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group("critical")
def critical_group():
    """Critical-set evaluation: metrics + per-recording timeline plots across every classifier."""


@critical_group.command("eval")
@click.option("--model", "-m", "models", multiple=True,
              help="Subset of models to evaluate (default: every model in config.toml).")
def eval_cmd(models: tuple[str, ...]):
    """Evaluate every (or selected) model on the crit test split."""
    cfg = _load_cfg()
    selected = list(models) if models else cfg["models"]
    override = _build_dataset_override(cfg)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info(
        "Critical eval — dataset url=%s name=%s, %d model(s)",
        override["url"], override["name"], len(selected),
    )
    ok, skipped = [], []
    for m in selected:
        if _eval_one(m, override):
            ok.append(m)
        else:
            skipped.append(m)

    log.info("\nSummary: %d evaluated, %d skipped", len(ok), len(skipped))
    if skipped:
        log.info("  skipped: %s", ", ".join(skipped))


@critical_group.command("visualize")
def visualize_cmd():
    """Render per-recording true-vs-predicted timeline SVGs from the saved `_scores.npz` files."""
    cfg = _load_cfg()
    override = _build_dataset_override(cfg)
    frame_ms_by_model: dict[str, float] = cfg["frame_ms"]
    viz = cfg["viz"]

    log.info("Loading clip metadata from %s ...", override["url"])
    clips = _load_clip_metadata(override["url"])
    log.info("Found %d test clip(s).", len(clips))

    per_model_data: dict[str, dict[str, np.ndarray]] = {}
    for model in cfg["models"]:
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

    if _GRAPHS_DIR.exists():
        for stale in _GRAPHS_DIR.glob("clip*.svg"):
            stale.unlink()
    _GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    written_total: list[Path] = []
    for clip_idx, meta in enumerate(clips):
        rows = _build_clip_rows(clip_idx, per_model_data, frame_ms_by_model)
        if len(rows) <= 1:  # only GT, no model rows
            log.warning("clip %d (%s): no model predictions → skipping", clip_idx, meta["source"])
            continue
        written = _render_clip(
            clip_idx, meta, rows, _GRAPHS_DIR,
            panel_seconds=float(viz["panel_seconds"]),
            max_panels_per_file=int(viz["max_panels_per_file"]),
            inches_per_second=float(viz["inches_per_second"]),
            row_height=float(viz["row_height"]),
        )
        for p in written:
            log.info("saved → %s", p)
        written_total.extend(written)

    log.info(
        "\nWrote %d plot file(s) for %d clip(s) into %s",
        len(written_total), len(clips), _GRAPHS_DIR,
    )


@critical_group.command("run-all")
@click.option("--model", "-m", "models", multiple=True,
              help="Subset of models to evaluate (default: every model in config.toml).")
@click.pass_context
def run_all_cmd(ctx, models):
    """Evaluate every model and render the timeline plots in one shot."""
    ctx.invoke(eval_cmd, models=models)
    ctx.invoke(visualize_cmd)
