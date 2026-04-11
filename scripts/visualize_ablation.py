"""Visualize TCN ablation study results grouped by subgroup.

Two modes:
  ofat         — OFAT study: ΔF1 heatmap + sensitivity ranking summary
  coord-ascent — Greedy search: trajectory line plot + config diff table summary
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Import parse_eval from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from visualize_results import parse_eval  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "src" / "exp" / "tcn_ablation" / "results"
EXP_CFG = _REPO_ROOT / "src" / "exp" / "tcn_ablation" / "config.toml"

_PALETTE = [
    "#2196F3", "#E91E63", "#FF9800", "#4CAF50",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]
_BASELINE_COLOR = "#90A4AE"
_DIVERGED_COLOR = "#EF5350"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_subgroups() -> dict[str, list[str]]:
    import tomli
    with open(EXP_CFG, "rb") as f:
        raw = tomli.load(f)
    ablations = raw.get("tcn", {}).get("ablations", {})
    return {grp: list(names.keys()) for grp, names in ablations.items()}


def _load_coord_ascent_order() -> list[str]:
    import tomli
    with open(EXP_CFG, "rb") as f:
        raw = tomli.load(f)
    return raw.get("tcn", {}).get("coord_ascent", {}).get("subgroup_order", [])


def _load_results(names: list[str], suffix: str = "") -> dict[str, dict]:
    """Load eval results for the given variant names.

    Variants with a `.diverged` sentinel are included as ``{"diverged": True}``.
    *suffix* is appended to the file stem (e.g. ``"_ca"`` for coord-ascent).
    """
    data = {}
    for name in names:
        eval_path = RESULTS_DIR / f"tcn_{name}{suffix}.eval"
        div_path  = RESULTS_DIR / f"tcn_{name}{suffix}.diverged"
        if eval_path.exists():
            data[name] = parse_eval(eval_path)
        elif div_path.exists():
            data[name] = {"diverged": True}
    return data


def _load_trajectory() -> dict | None:
    path = RESULTS_DIR / "coord_ascent_trajectory.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _is_diverged(data: dict, name: str) -> bool:
    return data.get(name, {}).get("diverged", False)


def _variant_color(name: str, index: int) -> str:
    if name == "baseline":
        return _BASELINE_COLOR
    return _PALETTE[index % len(_PALETTE)]


# ── Per-subgroup figure (shared by both modes) ────────────────────────────────

def _plot_subgroup(ax_f1, ax_class, subgroup: str, names: list[str], data: dict,
                   ref_f1: float | None = None):
    """Two-panel plot: macro F1 bars (left) + per-class F1 grouped bars (right).

    *ref_f1* overrides the baseline lookup for the reference line — used by
    coord-ascent to show the previous subgroup's winner F1 as the reference.
    """
    present = [n for n in names if n in data]
    if not present:
        ax_f1.set_visible(False)
        ax_class.set_visible(False)
        return

    colors = [_DIVERGED_COLOR if _is_diverged(data, n) else _variant_color(n, i)
              for i, n in enumerate(present)]
    x = np.arange(len(present))

    # Determine reference line: explicit ref_f1 > baseline in data > none
    if ref_f1 is None:
        ref_f1 = data.get("baseline", {}).get("macro_f1")

    # ── Macro F1 bars ──────────────────────────────────────────────────────
    f1_vals = [0.01 if _is_diverged(data, n) else (data[n].get("macro_f1") or 0)
               for n in present]
    bars = ax_f1.bar(x, f1_vals, 0.6, color=colors, alpha=0.88,
                     edgecolor=["black" if n == "baseline" else "none" for n in present],
                     linewidth=0.8)
    for bar, name in zip(bars, present):
        if _is_diverged(data, name):
            bar.set_hatch("//")
            bar.set_edgecolor(_DIVERGED_COLOR)
            bar.set_alpha(0.6)
            ax_f1.text(bar.get_x() + bar.get_width() / 2, 0.03,
                       "DIVERGED", ha="center", va="bottom",
                       fontsize=6, color=_DIVERGED_COLOR, rotation=90)
    if ref_f1 is not None:
        ax_f1.axhline(ref_f1, color=_BASELINE_COLOR, linestyle="--", linewidth=1.2,
                      label=f"ref {ref_f1:.3f}", zorder=0)
        ax_f1.legend(fontsize=7)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(present, rotation=20, ha="right", fontsize=8)
    valid_f1 = [v for n, v in zip(present, f1_vals) if not _is_diverged(data, n)]
    y_lo = max(0, min(valid_f1) - 0.05) if valid_f1 else 0
    ax_f1.set_ylim(y_lo, 1.02)
    ax_f1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_f1.set_title(f"{subgroup} — Macro F1", fontsize=10)
    ax_f1.set_ylabel("Macro F1")
    for bar, name, val in zip(bars, present, f1_vals):
        if not _is_diverged(data, name) and val > 0.01:
            ax_f1.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    # ── Per-class F1 grouped bars (diverged excluded) ──────────────────────
    classes = ["speech", "music", "inactive"]
    class_labels = ["Speech", "Music", "Inactive"]
    converged = [n for n in present if not _is_diverged(data, n)]
    n = len(converged)
    width = 0.22
    x2 = np.arange(len(classes))
    for i, name in enumerate(converged):
        pc = data[name].get("per_class", {})
        vals = [pc.get(c, {}).get("f1", 0) for c in classes]
        color = _variant_color(name, present.index(name))
        offset = (i - n / 2 + 0.5) * width
        ax_class.bar(x2 + offset, vals, width * 0.9, color=color, alpha=0.85,
                     label=name,
                     edgecolor="black" if name == "baseline" else "none",
                     linewidth=0.6)
    ax_class.set_xticks(x2)
    ax_class.set_xticklabels(class_labels, fontsize=9)
    ax_class.set_ylim(0, 1.05)
    ax_class.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_class.set_title(f"{subgroup} — Per-class F1", fontsize=10)
    ax_class.set_ylabel("F1")
    if converged:
        ax_class.legend(fontsize=6.5, ncol=max(1, n // 3))


def plot_subgroup_figure(subgroup: str, names: list[str], data: dict,
                         ref_f1: float | None = None) -> plt.Figure:
    fig, (ax_f1, ax_class) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Ablation: {subgroup}", fontsize=12, fontweight="bold")
    _plot_subgroup(ax_f1, ax_class, subgroup, names, data, ref_f1=ref_f1)
    fig.tight_layout()
    return fig


# ── OFAT summary ──────────────────────────────────────────────────────────────

def _plot_delta_heatmap(ax, subgroups: dict[str, list[str]], all_data: dict):
    """ΔF1 heatmap: rows=subgroups, cols=all non-baseline variants."""
    # Collect all non-baseline variant names in encounter order
    all_variants = list(dict.fromkeys(
        n for names in subgroups.values() for n in names if n != "baseline"
    ))
    subgroup_names = list(subgroups.keys())

    matrix = np.full((len(subgroup_names), len(all_variants)), np.nan)
    for r, grp in enumerate(subgroup_names):
        grp_names = subgroups[grp]
        baseline_f1 = all_data.get("baseline", {}).get("macro_f1")
        for c, var in enumerate(all_variants):
            if var not in grp_names:
                continue
            if _is_diverged(all_data, var):
                matrix[r, c] = np.nan  # handled separately below
            elif all_data.get(var, {}).get("macro_f1") is not None and baseline_f1 is not None:
                matrix[r, c] = all_data[var]["macro_f1"] - baseline_f1

    vmax = max(0.10, np.nanmax(np.abs(matrix))) if not np.all(np.isnan(matrix)) else 0.10
    cmap = plt.cm.RdYlGn
    cmap.set_bad("whitesmoke")
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="ΔF1 vs baseline", shrink=0.8)

    ax.set_xticks(range(len(all_variants)))
    ax.set_xticklabels(all_variants, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(subgroup_names)))
    ax.set_yticklabels(subgroup_names, fontsize=8)
    ax.set_title("ΔF1 vs baseline (OFAT)", fontsize=11)

    # Annotate cells
    for r, grp in enumerate(subgroup_names):
        grp_names = subgroups[grp]
        baseline_f1 = all_data.get("baseline", {}).get("macro_f1")
        for c, var in enumerate(all_variants):
            if var not in grp_names:
                continue
            if _is_diverged(all_data, var):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           hatch="//", fill=False,
                                           edgecolor=_DIVERGED_COLOR, linewidth=0))
                ax.text(c, r, "✕", ha="center", va="center",
                        fontsize=9, color=_DIVERGED_COLOR, fontweight="bold")
            elif not np.isnan(matrix[r, c]):
                val = matrix[r, c]
                txt = f"{val:+.3f}"
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(c, r, txt, ha="center", va="center", fontsize=7, color=color)


def _plot_sensitivity(ax, subgroups: dict[str, list[str]], all_data: dict):
    """Horizontal bar: F1 range within each subgroup (sensitivity ranking)."""
    rows = []
    for grp, names in subgroups.items():
        f1s = [all_data[n]["macro_f1"] for n in names
               if n in all_data and not _is_diverged(all_data, n)
               and all_data[n].get("macro_f1") is not None]
        if len(f1s) < 2:
            continue
        spread = max(f1s) - min(f1s)
        winner = max((n for n in names if n in all_data and not _is_diverged(all_data, n)
                      and all_data[n].get("macro_f1") is not None),
                     key=lambda n: all_data[n]["macro_f1"])
        rows.append((grp, spread, winner))

    if not rows:
        ax.set_visible(False)
        return

    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    spreads = [r[1] for r in rows]
    winners = [r[2] for r in rows]
    y = np.arange(len(rows))

    bars = ax.barh(y, spreads, 0.6, color="#42A5F5", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("F1 range (max − min)", fontsize=8)
    ax.set_title("Sensitivity Ranking", fontsize=11)
    for bar, spread, winner in zip(bars, spreads, winners):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"+{spread:.3f}  [{winner}]", va="center", fontsize=7, color="#555")
    ax.set_xlim(0, max(spreads) * 1.55)


def plot_ofat_summary(subgroups: dict[str, list[str]], all_data: dict) -> plt.Figure:
    """OFAT summary: ΔF1 heatmap (left) + sensitivity ranking (right)."""
    fig, (ax_heat, ax_sens) = plt.subplots(
        1, 2, figsize=(18, max(5, len(subgroups) * 0.9)),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle("OFAT Ablation Study — Summary", fontsize=13, fontweight="bold")
    _plot_delta_heatmap(ax_heat, subgroups, all_data)
    _plot_sensitivity(ax_sens, subgroups, all_data)
    fig.tight_layout()
    return fig


# ── Coord-ascent summary ──────────────────────────────────────────────────────

def _plot_trajectory(ax, trajectory: dict, subgroup_order: list[str], baseline_f1: float | None):
    """Line plot of winner F1 through the coordinate-ascent subgroups."""
    present = [g for g in subgroup_order if g in trajectory]
    if not present:
        ax.set_visible(False)
        return

    xs = list(range(len(present)))
    f1s = [trajectory[g]["f1"] for g in present]
    winners = [trajectory[g]["winner"] for g in present]

    if baseline_f1 is not None:
        ax.axhline(baseline_f1, color=_BASELINE_COLOR, linestyle=":", linewidth=1.5,
                   label=f"fixed baseline {baseline_f1:.3f}", zorder=0)
        ax.fill_between(xs, baseline_f1, f1s, alpha=0.12, color="#2196F3")

    ax.plot(xs, f1s, "o-", color="#2196F3", linewidth=2, markersize=8, zorder=3)
    for x, f1, winner in zip(xs, f1s, winners):
        ax.annotate(f"{winner}\n{f1:.3f}", (x, f1),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color="#1565C0")

    ax.set_xticks(xs)
    ax.set_xticklabels(present, rotation=25, ha="right", fontsize=8)
    y_lo = min(f1s + ([baseline_f1] if baseline_f1 else [])) - 0.02
    ax.set_ylim(max(0, y_lo), 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylabel("Macro F1")
    ax.set_title("Coord-Ascent Trajectory", fontsize=11)
    ax.legend(fontsize=8)


def _plot_config_diff(ax, trajectory: dict, subgroup_order: list[str]):
    """Table showing: subgroup | winner | ΔF1 | changed params."""
    present = [g for g in subgroup_order if g in trajectory]
    if not present:
        ax.set_visible(False)
        return

    ax.axis("off")
    col_labels = ["Subgroup", "Winner", "ΔF1", "Changed params"]
    rows = []
    prev_f1 = None
    prev_carried: dict = {}

    for grp in present:
        entry = trajectory[grp]
        winner = entry["winner"]
        f1 = entry["f1"]
        carried = entry.get("carried", {})

        delta = f"{f1 - prev_f1:+.4f}" if prev_f1 is not None else "—"

        changed = {k: v for k, v in carried.items() if prev_carried.get(k) != v}
        changed_str = ", ".join(f"{k}={v}" for k, v in changed.items()) or "—"

        rows.append([grp, winner, delta, changed_str])
        prev_f1 = f1
        prev_carried = carried

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Zebra stripes on data rows
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#E3F2FD" if i % 2 == 0 else "white")

    ax.set_title("Config Changes per Step", fontsize=11, pad=14)


def plot_coord_ascent_summary(subgroups: dict, all_data: dict, trajectory: dict,
                               subgroup_order: list[str]) -> plt.Figure:
    """Coord-ascent summary: trajectory line (top) + config diff table (bottom)."""
    fig, (ax_traj, ax_diff) = plt.subplots(2, 1, figsize=(14, 10),
                                            gridspec_kw={"height_ratios": [1, 1.4]})
    fig.suptitle("Coordinate-Ascent Study — Summary", fontsize=13, fontweight="bold")

    baseline_f1 = all_data.get("baseline", {}).get("macro_f1")
    _plot_trajectory(ax_traj, trajectory, subgroup_order, baseline_f1)
    _plot_config_diff(ax_diff, trajectory, subgroup_order)

    fig.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main(subgroup_filter: str | None = None, suffix: str = ""):
    subgroups = _load_subgroups()
    coord_order = _load_coord_ascent_order()

    all_names = list(dict.fromkeys(n for names in subgroups.values() for n in names))
    all_data = _load_results(all_names, suffix=suffix)

    trajectory = _load_trajectory() if suffix == "_ca" else None

    if not all_data:
        print(f"No eval files found in results/ (suffix={suffix!r}). Run the ablation first.")
        return

    graphs_dir = RESULTS_DIR / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    groups_to_plot = (
        {subgroup_filter: subgroups[subgroup_filter]}
        if subgroup_filter and subgroup_filter in subgroups
        else subgroups
    )

    # Per-subgroup figures
    prev_f1: float | None = None
    for subgroup, names in groups_to_plot.items():
        data = {n: all_data[n] for n in names if n in all_data}
        if not data:
            print(f"[{subgroup}] no eval files found — skipping")
            continue

        # Coord-ascent: use previous subgroup's winner F1 as reference line
        ref_f1 = None
        if suffix == "_ca" and trajectory:
            prev_grps = [g for g in (coord_order or list(subgroups)) if g != subgroup]
            for g in reversed(prev_grps):
                if g in trajectory:
                    ref_f1 = trajectory[g]["f1"]
                    break

        fig = plot_subgroup_figure(subgroup, names, data, ref_f1=ref_f1)
        out = graphs_dir / f"ablation_{subgroup}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")

    if subgroup_filter:
        return

    # Summary figure
    if suffix == "_ca" and trajectory:
        fig = plot_coord_ascent_summary(subgroups, all_data, trajectory, coord_order)
        out = graphs_dir / "ablation_ca_summary.png"
    else:
        fig = plot_ofat_summary(subgroups, all_data)
        out = graphs_dir / "ablation_summary.png"

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
