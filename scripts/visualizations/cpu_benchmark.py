"""Single-thread CPU benchmark — quality vs cost trade-off scatter.

Replaces / complements tab:complexity_t1_results in ch6
§sec:exp_complexity_results.

Plot rationale:
  The table reports F1, RTF, and memory across all seven deployed
  models. The deployment recommendation chapter then re-reads those
  numbers as a trade-off: pick by which axis binds. The trade-off
  is much more legible as a scatter where x = real-time factor,
  y = test-split macro F1, and marker size encodes resident memory.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from _common import COLORS, MODEL_SHORT, save_svg, setup_style


def _drss_to_size(drss_mb: float) -> float:
    """Map marginal RSS (MB) to scatter marker area (points^2)."""
    return 30 + 8 * (drss_mb ** 0.5)

# From src/exp/complexity/results/complexity_t1.md (single-thread CPU run).
# Test-split macro F1 from results/<model>.eval.
ROWS = [
    # name,             RTF,  test_F1, dRSS_MB
    ("tcn_l",           2.1,  0.9846,  23),
    ("tcn",             3.2,  0.9752,  8),
    ("tcn_s",           5.3,  0.9722,  13),
    ("svm",             2.0,  0.8750,  9),
    ("decision_tree",   2.4,  0.8376,  368),
    ("gmm",             3.8,  0.8250,  3),
]


def main():
    setup_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    for name, rtf, f1, drss in ROWS:
        # Bubble area scaled with sqrt(memory) so 368 MB doesn't swamp 3 MB.
        color = COLORS[name]
        ax.scatter(
            rtf, f1, s=_drss_to_size(drss), color=color, alpha=0.8,
            edgecolor="black", lw=0.7, zorder=3,
        )
        # Label slightly offset.
        offset_y = 0.005 if name not in ("svm", "decision_tree") else -0.012
        ax.annotate(
            f"{MODEL_SHORT[name]}\n({drss} MB)",
            xy=(rtf, f1),
            xytext=(0, 14 if offset_y > 0 else -22), textcoords="offset points",
            ha="center", fontsize=7,
            color=color,
        )

    # Real-time floor at 1×.
    ax.axvline(1.0, ls=":", color="red", lw=0.8, alpha=0.5)

    ax.set_xlabel(r"Real-time factor $\quad$ better $\rightarrow$")
    ax.set_ylabel(r"Macro F1 $\quad$ better $\rightarrow$")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(1.7, 6.0)
    ax.set_ylim(0.80, 1.00)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="lightgray", markeredgecolor="black",
            markeredgewidth=0.6, markersize=_drss_to_size(50) ** 0.5,
            label=r"$\Delta$RSS (smaller $=$ better)",
        )
    ]
    ax.legend(
        handles=handles,
        loc="lower right", fontsize=7,
        frameon=True, framealpha=0.9,
        borderpad=0.7, handletextpad=1.0,
    )

    out = save_svg(fig, "cpu_benchmark")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
