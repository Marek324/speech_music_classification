# scripts/visualizations/transitions_latency.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Two-class switching latency heatmap (median ms by model × cadence).

Replaces tab:transitions_2class_latency in ch6 §sec:exp_transitions_results.

Plot rationale:
  The current table is 7 models × 4 cadences with three numbers per cell
  (median, p90, miss%) — dense and hard to skim. A heatmap of median
  latency reads directly: dark cells are slow models / cadences, light
  cells are fast. Miss-rate annotations stay as text so the reader
  doesn't lose the secondary metric.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import save_svg, setup_style

CADENCES = [500, 1000, 2000, 4000]
MODELS = ["DT", "TCN-L", "TCN", "TCN-S", "GMM", "SVM"]
LATENCY = {
    "TCN":        [46, 70, 139, 70],
    "TCN-L":      [0, 0, 0, 23],
    "TCN-S":      [46, 46, 46, 23],
    "DT":         [30, 40, 30, 35],
    "GMM":        [0, 210, 180, 180],
    "SVM":        [0, 322, 345, 345],
}
MISS_RATE = {
    "TCN":        [12, 5, 4, 6],
    "TCN-L":      [40, 29, 0, 6],
    "TCN-S":      [3, 0, 0, 0],
    "DT":         [3, 0, 0, 0],
    "GMM":        [44, 13, 9, 6],
    "SVM":        [51, 16, 22, 25],
}


def main():
    setup_style()
    matrix = np.array([LATENCY[m] for m in MODELS], dtype=float)
    miss = np.array([MISS_RATE[m] for m in MODELS], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    im = ax.imshow(
        matrix, cmap="YlOrRd", aspect="auto",
        vmin=0, vmax=350,
    )

    for i, _ in enumerate(MODELS):
        for j, _ in enumerate(CADENCES):
            v = matrix[i, j]
            m = miss[i, j]
            txt_color = "white" if v > 200 else "black"
            ax.text(
                j, i - 0.15, f"{int(v)} ms",
                ha="center", va="center", color=txt_color, fontsize=8,
            )
            if m > 0:
                ax.text(
                    j, i + 0.18, f"miss {int(m)}%",
                    ha="center", va="center", color=txt_color,
                    fontsize=6.5, alpha=0.85,
                )

    ax.set_xticks(range(len(CADENCES)))
    ax.set_xticklabels([f"{c} ms" for c in CADENCES])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("Switching cadence")
    ax.set_ylabel("Model")
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Median latency (ms)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out = save_svg(fig, "transitions_latency")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
