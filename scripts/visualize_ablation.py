"""Visualize TCN ablation study results grouped by subgroup."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Import parse_eval from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from visualize_results import parse_eval  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results"
EXP_CFG = _REPO_ROOT / "src" / "exp" / "tcn_ablation" / "config.toml"

# Color palette for variants (cycles if more than 8)
_PALETTE = [
    "#2196F3", "#E91E63", "#FF9800", "#4CAF50",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]
_BASELINE_COLOR = "#90A4AE"


def _load_subgroups() -> dict[str, list[str]]:
    """Read subgroup → [variant_names] from exp config.toml."""
    import tomli
    with open(EXP_CFG, "rb") as f:
        raw = tomli.load(f)
    ablations = raw.get("tcn", {}).get("ablations", {})
    return {grp: list(names.keys()) for grp, names in ablations.items()}


def _load_results(names: list[str]) -> dict[str, dict]:
    """Load eval results for the given variant names. Missing files are skipped."""
    data = {}
    for name in names:
        path = RESULTS_DIR / f"tcn_{name}.eval"
        if path.exists():
            data[name] = parse_eval(path)
    return data


def _variant_color(name: str, index: int) -> str:
    if name == "baseline":
        return _BASELINE_COLOR
    return _PALETTE[index % len(_PALETTE)]


def _plot_subgroup(ax_f1, ax_class, subgroup: str, names: list[str], data: dict):
    """Two-panel plot for one subgroup: macro F1 bars + per-class F1 grouped bars."""
    present = [n for n in names if n in data]
    if not present:
        ax_f1.set_visible(False)
        ax_class.set_visible(False)
        return

    colors = [_variant_color(n, i) for i, n in enumerate(present)]
    x = np.arange(len(present))
    baseline_f1 = data.get("baseline", {}).get("macro_f1")

    # ── Macro F1 bars ──────────────────────────────────────────────────────
    f1_vals = [data[n].get("macro_f1") or 0 for n in present]
    bars = ax_f1.bar(x, f1_vals, 0.6, color=colors, alpha=0.88,
                     edgecolor=["black" if n == "baseline" else "none" for n in present],
                     linewidth=0.8)
    if baseline_f1 is not None:
        ax_f1.axhline(baseline_f1, color=_BASELINE_COLOR, linestyle="--", linewidth=1.2,
                      label=f"baseline {baseline_f1:.3f}", zorder=0)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(present, rotation=20, ha="right", fontsize=8)
    ax_f1.set_ylim(max(0, min(f1_vals) - 0.05), 1.02)
    ax_f1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_f1.set_title(f"{subgroup} — Macro F1", fontsize=10)
    ax_f1.set_ylabel("Macro F1")
    for bar, val in zip(bars, f1_vals):
        if val > 0.01:
            ax_f1.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    # ── Per-class F1 grouped bars ───────────────────────────────────────────
    classes = ["speech", "music", "inactive"]
    class_labels = ["Speech", "Music", "Inactive"]
    n = len(present)
    width = 0.22
    x2 = np.arange(len(classes))
    for i, name in enumerate(present):
        pc = data[name].get("per_class", {})
        vals = [pc.get(c, {}).get("f1", 0) for c in classes]
        offset = (i - n / 2 + 0.5) * width
        ax_class.bar(x2 + offset, vals, width * 0.9, color=colors[i], alpha=0.85,
                     label=name,
                     edgecolor="black" if name == "baseline" else "none",
                     linewidth=0.6)
    ax_class.set_xticks(x2)
    ax_class.set_xticklabels(class_labels, fontsize=9)
    ax_class.set_ylim(0, 1.05)
    ax_class.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_class.set_title(f"{subgroup} — Per-class F1", fontsize=10)
    ax_class.set_ylabel("F1")
    ax_class.legend(fontsize=6.5, ncol=max(1, n // 3))


def plot_subgroup_figure(subgroup: str, names: list[str], data: dict) -> plt.Figure:
    fig, (ax_f1, ax_class) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Ablation: {subgroup}", fontsize=12, fontweight="bold")
    _plot_subgroup(ax_f1, ax_class, subgroup, names, data)
    fig.tight_layout()
    return fig


def plot_summary(subgroups: dict[str, list[str]], all_data: dict[str, dict]) -> plt.Figure:
    """One panel per subgroup showing macro F1 bars + baseline reference line."""
    n_groups = len(subgroups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4.5), squeeze=False)
    fig.suptitle("Ablation Study — Macro F1 Summary", fontsize=13, fontweight="bold")

    for ax, (subgroup, names) in zip(axes[0], subgroups.items()):
        present = [n for n in names if n in all_data]
        if not present:
            ax.set_visible(False)
            continue
        colors = [_variant_color(n, i) for i, n in enumerate(present)]
        x = np.arange(len(present))
        f1_vals = [all_data[n].get("macro_f1") or 0 for n in present]
        baseline_f1 = all_data.get("baseline", {}).get("macro_f1")

        bars = ax.bar(x, f1_vals, 0.65, color=colors, alpha=0.88,
                      edgecolor=["black" if n == "baseline" else "none" for n in present],
                      linewidth=0.8)
        if baseline_f1 is not None:
            ax.axhline(baseline_f1, color=_BASELINE_COLOR, linestyle="--", linewidth=1.1, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=25, ha="right", fontsize=7.5)
        y_lo = max(0, min(f1_vals) - 0.06) if f1_vals else 0
        ax.set_ylim(y_lo, 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_title(subgroup, fontsize=10)
        if ax is axes[0][0]:
            ax.set_ylabel("Macro F1")
        for bar, val in zip(bars, f1_vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    return fig


def main(subgroup_filter: str | None = None):
    subgroups = _load_subgroups()

    # Collect unique variant names
    all_names = list(dict.fromkeys(n for names in subgroups.values() for n in names))
    all_data = _load_results(all_names)

    if not all_data:
        print("No eval files found in results/. Run ablation first.")
        return

    graphs_dir = RESULTS_DIR / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    groups_to_plot = (
        {subgroup_filter: subgroups[subgroup_filter]}
        if subgroup_filter and subgroup_filter in subgroups
        else subgroups
    )

    for subgroup, names in groups_to_plot.items():
        data = {n: all_data[n] for n in names if n in all_data}
        if not data:
            print(f"[{subgroup}] no eval files found — skipping")
            continue
        fig = plot_subgroup_figure(subgroup, names, data)
        out = graphs_dir / f"ablation_{subgroup}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")

    if not subgroup_filter:
        fig = plot_summary(subgroups, all_data)
        out = graphs_dir / "ablation_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
