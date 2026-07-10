# scripts/visualizations/tcn_dilation.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Figure 3.2: dilated convolutions, receptive-field illustration.

Three layers with dilations 1, 2, 4 and centered kernel size 3.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from _common import REPO_ROOT, setup_style

OUT_DIR = REPO_ROOT / "thesis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 3
DILATIONS = [1, 2, 4]
K = 2
RF = 1 + (K - 1) * sum(DILATIONS)
N_COLS = RF

BG_INPUT_COLOR = "#CFD8DC"
BG_HIDDEN_COLOR = "#FFFFFF"
BG_EDGE = "#B0BEC5"
RF_INPUT = "#2196F3"
RF_HIDDEN = "#FFC107"
RF_OUTPUT = "#E91E63"
HIGHLIGHT_LINE = "#E91E63"


def save(fig: Figure, name: str) -> Path:
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


def render(out_name: str) -> Path:
    fig: Figure = Figure(figsize=(6.5, 2.2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    for layer in range(N_LAYERS + 1):
        for col in range(N_COLS):
            if layer == 0:
                ax.add_patch(Circle(
                    (col, layer), 0.18,
                    facecolor=BG_INPUT_COLOR, edgecolor="none", zorder=1,
                ))
            else:
                ax.add_patch(Circle(
                    (col, layer), 0.18,
                    facecolor=BG_HIDDEN_COLOR, edgecolor=BG_EDGE,
                    linewidth=0.5, zorder=1,
                ))

    center = N_COLS - 1
    rf_nodes = {(N_LAYERS, center)}
    rf_edges: set[tuple[int, int, int, int]] = set()
    queue = [(N_LAYERS, center)]
    while queue:
        layer, col = queue.pop()
        if layer == 0:
            continue
        d = DILATIONS[layer - 1]
        for offset in range(K):
            child_col = col - offset * d
            if 0 <= child_col < N_COLS:
                rf_edges.add((layer, col, layer - 1, child_col))
                child = (layer - 1, child_col)
                if child not in rf_nodes:
                    rf_nodes.add(child)
                    queue.append(child)

    for layer in range(1, N_LAYERS + 1):
        d = DILATIONS[layer - 1]
        for col in range(N_COLS):
            for offset in range(K):
                child_col = col - offset * d
                if 0 <= child_col < N_COLS:
                    edge = (layer, col, layer - 1, child_col)
                    if edge in rf_edges:
                        continue
                    ax.plot([col, child_col], [layer, layer - 1],
                            color="#CFD8DC", linewidth=0.5, zorder=1, alpha=0.6)

    for lp, cp, lc, cc in rf_edges:
        ax.plot([cp, cc], [lp, lc],
                color=HIGHLIGHT_LINE, linewidth=0.7, zorder=2, alpha=0.55)

    for layer, col in rf_nodes:
        if layer == 0:
            color = RF_INPUT
        elif layer == N_LAYERS:
            color = RF_OUTPUT
        else:
            color = RF_HIDDEN
        ax.add_patch(Circle(
            (col, layer), 0.22,
            facecolor=color, edgecolor="black", linewidth=0.5, zorder=3,
        ))

    for layer in range(1, N_LAYERS + 1):
        d = DILATIONS[layer - 1]
        ax.text(N_COLS + 0.5, layer, f"$d = {d}$",
                ha="left", va="center", fontsize=9, color="#444")

    ax.text(-1.0, 0, "input", ha="right", va="center", fontsize=9, color="#444")
    for layer in range(1, N_LAYERS + 1):
        ax.text(-1.0, layer, f"layer {layer}",
                ha="right", va="center", fontsize=9, color="#444")

    ax.set_xlim(-3.0, N_COLS + 3.5)
    ax.set_ylim(-0.6, N_LAYERS + 0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()

    return save(fig, out_name)


def main() -> None:
    setup_style()
    out = render(out_name="tcn_dilation")
    print(f"Wrote {out} (+ .svg)")


if __name__ == "__main__":
    main()
