"""Visualize flat-variant experiment results (frontend, architecture, preprocessor).

Produces a single PNG with 2–3 panels:
  1. Macro F1 bar chart (one bar per variant)
  2. Per-class F1 grouped bars (Speech / Music / Inactive)
  3. CI whisker plot (only if bootstrap CIs exist in .eval files)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Import parse_eval from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from visualize_results import parse_eval  # noqa: E402

_PALETTE = [
    "#2196F3", "#E91E63", "#FF9800", "#4CAF50",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]
_BASELINE_COLOR = "#90A4AE"
_DIVERGED_COLOR = "#EF5350"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_variants(config_path: Path) -> list[str]:
    """Read variant names from [tcn.variants] in experiment config.toml."""
    import tomli
    with open(config_path, "rb") as f:
        raw = tomli.load(f)
    return list(raw.get("tcn", {}).get("variants", {}).keys())


def load_results(results_dir: Path, names: list[str]) -> dict[str, dict]:
    """Load eval results for the given variant names.

    Variants with a ``.diverged`` sentinel are included as ``{"diverged": True}``.
    """
    data = {}
    for name in names:
        eval_path = results_dir / f"tcn_{name}.eval"
        div_path = results_dir / f"tcn_{name}.diverged"
        if eval_path.exists():
            data[name] = parse_eval(eval_path)
        elif div_path.exists():
            data[name] = {"diverged": True}
    return data


def _is_diverged(data: dict, name: str) -> bool:
    return data.get(name, {}).get("diverged", False)


def _variant_color(name: str, index: int) -> str:
    if name == "baseline":
        return _BASELINE_COLOR
    return _PALETTE[index % len(_PALETTE)]


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_macro_f1(ax, names: list[str], data: dict):
    """Macro F1 bar chart with value annotations and diverged markers."""
    present = [n for n in names if n in data]
    if not present:
        ax.set_visible(False)
        return

    colors = [_DIVERGED_COLOR if _is_diverged(data, n) else _variant_color(n, i)
              for i, n in enumerate(present)]
    x = np.arange(len(present))

    ref_f1 = data.get("baseline", {}).get("macro_f1")

    valid_f1 = [data[n]["macro_f1"] for n in present
                if not _is_diverged(data, n) and data[n].get("macro_f1")]
    y_lo = max(0, min(valid_f1) - 0.05) if valid_f1 else 0

    f1_vals = [0 if _is_diverged(data, n) else (data[n].get("macro_f1") or 0)
               for n in present]
    bars = ax.bar(x, f1_vals, 0.6, color=colors, alpha=0.88,
                  edgecolor=["black" if n == "baseline" else "none" for n in present],
                  linewidth=0.8)

    # Restyle diverged bars
    div_bar_h = (1.0 - y_lo) * 0.38
    for bar, name in zip(bars, present):
        if _is_diverged(data, name):
            bar.set_y(y_lo)
            bar.set_height(div_bar_h)
            bar.set_facecolor(_DIVERGED_COLOR)
            bar.set_alpha(0.35)
            bar.set_hatch("///")
            bar.set_edgecolor(_DIVERGED_COLOR)
            bar.set_linewidth(0.5)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_lo + div_bar_h / 2,
                    "DIVERGED", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="#B71C1C", zorder=5)

    if ref_f1 is not None:
        ax.axhline(ref_f1, color=_BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"baseline {ref_f1:.3f}", zorder=0)
        ax.legend(fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(y_lo, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Macro F1", fontsize=10)
    ax.set_ylabel("Macro F1")

    for bar, name, val in zip(bars, present, f1_vals):
        if not _is_diverged(data, name) and val > 0.01:
            inside = val > 0.975
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val - 0.005 if inside else val + 0.003,
                    f"{val:.3f}", ha="center",
                    va="top" if inside else "bottom",
                    fontsize=6.5,
                    color="white" if inside else "black")


def _plot_per_class_f1(ax, names: list[str], data: dict):
    """Per-class F1 grouped bar chart."""
    classes = ["speech", "music", "inactive"]
    class_labels = ["Speech", "Music", "Inactive"]
    converged = [n for n in names if n in data and not _is_diverged(data, n)]
    n = len(converged)
    if not n:
        ax.set_visible(False)
        return

    width = min(0.22, 0.75 / n)
    x = np.arange(len(classes))
    for i, name in enumerate(converged):
        pc = data[name].get("per_class", {})
        vals = [pc.get(c, {}).get("f1", 0) for c in classes]
        color = _variant_color(name, names.index(name))
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, color=color, alpha=0.85,
               label=name,
               edgecolor="black" if name == "baseline" else "none",
               linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Per-class F1", fontsize=10)
    ax.set_ylabel("F1")
    ax.legend(fontsize=6.5, ncol=max(1, n // 3))


def _plot_ci_whiskers(ax, names: list[str], data: dict):
    """Dot-and-whisker plot of macro F1 with 95% bootstrap CI."""
    converged = [n for n in names if n in data and not _is_diverged(data, n)
                 and data[n].get("macro_f1_ci")]
    if not converged:
        ax.set_visible(False)
        return

    y = np.arange(len(converged))
    f1s = [data[n]["macro_f1"] for n in converged]
    lo = [data[n]["macro_f1"] - data[n]["macro_f1_ci"][0] for n in converged]
    hi = [data[n]["macro_f1_ci"][1] - data[n]["macro_f1"] for n in converged]
    colors = [_variant_color(n, names.index(n)) for n in converged]

    ax.errorbar(f1s, y, xerr=[lo, hi], fmt="o", capsize=4,
                markersize=6, elinewidth=1.5, capthick=1.2,
                color="#333")
    for yi, f1, c in zip(y, f1s, colors):
        ax.plot(f1, yi, "o", color=c, markersize=8, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(converged, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_title("Macro F1 — 95% Bootstrap CI", fontsize=10)
    ax.set_xlabel("Macro F1")


def _has_any_ci(names: list[str], data: dict) -> bool:
    return any(
        data.get(n, {}).get("macro_f1_ci")
        for n in names if n in data and not _is_diverged(data, n)
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def main(config_path: Path, results_dir: Path, experiment_name: str):
    names = load_variants(config_path)
    data = load_results(results_dir, names)

    if not data:
        print(f"No eval files found in {results_dir}. Run the experiment first.")
        return

    graphs_dir = results_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    show_ci = _has_any_ci(names, data)
    ncols = 3 if show_ci else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), constrained_layout=True)
    fig.suptitle(experiment_name, fontsize=12, fontweight="bold")

    _plot_macro_f1(axes[0], names, data)
    _plot_per_class_f1(axes[1], names, data)
    if show_ci:
        _plot_ci_whiskers(axes[2], names, data)

    slug = experiment_name.lower().replace(" ", "_")
    out = graphs_dir / f"{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
