"""Symbolic complexity — params vs MACs/frame log-log scatter.

Complements tab:complexity_symbolic in ch5 §subsec:exp_complexity_symbolic_results.

Plot rationale:
  The table reports six symbolic quantities per model; the two that
  span the widest range and decouple most cleanly across the family
  are storage cost (parameters) and streaming compute (MACs per output
  frame). DT sits in the high-params / low-MACs corner (large tree,
  trivial compute), the TCN family sits in the low-params / high-MACs
  corner (tiny weights, receptive-field re-run on every push), and
  the classics span the diagonal between them. Marker area encodes
  test-split macro F1 so the figure also answers "is the cost justified?".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from _common import COLORS, MODEL_SHORT, save_svg, setup_style

# From src/exp/complexity/results/complexity_t1.md, matching tab:complexity_symbolic.
# F1 from results/<model>.eval.
ROWS = [
    # name,           params,    MACs/frame,   F1
    ("tcn_l",         388_900,   137_260_000,  0.9846),
    ("tcn",            32_800,    11_570_000,  0.9752),
    ("tcn_s",           4_600,       545_000,  0.9722),
    ("svm",           520_800,       468_700,  0.8750),
    ("decision_tree", 4_340_000,         178,  0.8376),
    ("gmm",                474,           216,  0.8250),
]

LABEL_OFFSETS = {
    "tcn_l":         (14, -4),
    "tcn":           (14, -4),
    "tcn_s":         (14, -4),
    "svm":           (12, -4),
    "decision_tree": (-12, -14),
    "gmm":           (10, 4),
}

LABEL_HA = {
    "decision_tree": "right",
}


def _f1_to_size(f1: float) -> float:
    """Map macro F1 in [0.80, 1.00] to scatter marker area (points^2)."""
    return 30 + max(0.0, f1 - 0.80) * 2200


def main():
    setup_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    for name, params, macs, f1 in ROWS:
        color = COLORS[name]
        ax.scatter(
            params, macs, s=_f1_to_size(f1), color=color, alpha=0.85,
            edgecolor="black", lw=0.7, zorder=3,
        )
        dx, dy = LABEL_OFFSETS.get(name, (8, 4))
        ax.annotate(
            MODEL_SHORT[name],
            xy=(params, macs),
            xytext=(dx, dy), textcoords="offset points",
            ha=LABEL_HA.get(name, "left"),
            fontsize=8, color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\leftarrow$ better $\quad$ Parameter count")
    ax.set_ylabel(r"$\leftarrow$ better $\quad$ MACs per frame")
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.set_xlim(2e2, 1e7)
    ax.set_ylim(50, 5e8)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="lightgray", markeredgecolor="black",
            markeredgewidth=0.6, markersize=_f1_to_size(0.90) ** 0.5,
            label=r"Macro F1 (larger $=$ better)",
        )
    ]
    ax.legend(
        handles=handles,
        loc="upper left", fontsize=7,
        frameon=True, framealpha=0.9,
        borderpad=0.7, handletextpad=1.0,
    )

    out = save_svg(fig, "symbolic_complexity")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
