# scripts/visualizations/tcn_rf_blocks.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Per-block receptive-field illustration: 3 residual blocks at K=2,
dilations 1, 2, 4. Each block has 2 convs at the same dilation, so the
receptive field doubles compared to the one-conv-per-layer case shown in
fig:tcn_dilation.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from _common import REPO_ROOT, setup_style

OUT_DIR = REPO_ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 2
DILATIONS = [1, 1, 2, 2, 4, 4]
N_LAYERS = len(DILATIONS)
RF = 1 + 2 * (K - 1) * (2**3 - 1)
N_COLS = RF

INPUT_BG = "#CFD8DC"
HIDDEN_BG = "#FFFFFF"
HIDDEN_EDGE = "#B0BEC5"
RF_INPUT = "#2196F3"
RF_HIDDEN = "#FFC107"
RF_OUTPUT = "#E91E63"
HIGHLIGHT_LINE = "#E91E63"
BLOCK_TINTS = ["#F5F5F5", "#ECEFF1", "#F5F5F5"]
SPAN_COLOR = "#444"


def save(fig: Figure, name: str) -> Path:
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


def render(out_name: str) -> Path:
    fig: Figure = Figure(figsize=(11.5, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    for block_idx in range(3):
        bottom_layer = 1 + 2 * block_idx
        top_layer = 2 + 2 * block_idx
        ax.add_patch(Rectangle(
            (-0.5, bottom_layer - 0.42),
            N_COLS,
            (top_layer - bottom_layer) + 0.84,
            facecolor=BLOCK_TINTS[block_idx],
            edgecolor="#DDDDDD", linewidth=0.5, zorder=0,
        ))

    for layer in range(N_LAYERS + 1):
        for col in range(N_COLS):
            if layer == 0:
                ax.add_patch(Circle((col, layer), 0.18,
                                    facecolor=INPUT_BG, edgecolor="none",
                                    zorder=1))
            else:
                ax.add_patch(Circle((col, layer), 0.18,
                                    facecolor=HIDDEN_BG, edgecolor=HIDDEN_EDGE,
                                    linewidth=0.5, zorder=1))

    output_col = N_COLS - 1
    rf = {(N_LAYERS, output_col)}
    rf_edges: set[tuple[int, int, int, int]] = set()
    queue = [(N_LAYERS, output_col)]
    while queue:
        layer, col = queue.pop()
        if layer == 0:
            continue
        d = DILATIONS[layer - 1]
        for k in range(K):
            child_col = col - k * d
            if 0 <= child_col < N_COLS:
                rf_edges.add((layer, col, layer - 1, child_col))
                child = (layer - 1, child_col)
                if child not in rf:
                    rf.add(child)
                    queue.append(child)

    for layer in range(1, N_LAYERS + 1):
        d = DILATIONS[layer - 1]
        for col in range(N_COLS):
            for k in range(K):
                child_col = col - k * d
                if 0 <= child_col < N_COLS:
                    edge = (layer, col, layer - 1, child_col)
                    if edge in rf_edges:
                        continue
                    ax.plot([col, child_col], [layer, layer - 1],
                            color="#CFD8DC", linewidth=0.4, zorder=1, alpha=0.5)

    for lp, cp, lc, cc in rf_edges:
        ax.plot([cp, cc], [lp, lc],
                color=HIGHLIGHT_LINE, linewidth=0.6, zorder=2, alpha=0.45)

    for layer, col in rf:
        if layer == 0:
            color = RF_INPUT
        elif layer == N_LAYERS:
            color = RF_OUTPUT
        else:
            color = RF_HIDDEN
        ax.add_patch(Circle((col, layer), 0.22,
                            facecolor=color, edgecolor="black",
                            linewidth=0.5, zorder=3))

    for layer in range(1, N_LAYERS + 1):
        d = DILATIONS[layer - 1]
        ax.text(N_COLS + 0.4, layer, f"$d = {d}$",
                ha="left", va="center", fontsize=9, color="#444")

    for block_idx in range(3):
        bottom_layer = 1 + 2 * block_idx
        top_layer = 2 + 2 * block_idx
        center_y = (bottom_layer + top_layer) / 2
        ax.text(N_COLS + 2.0, center_y, f"block {block_idx + 1}",
                ha="left", va="center", fontsize=9, color="#444",
                fontweight="bold")
        ax.plot([N_COLS + 1.7, N_COLS + 1.85, N_COLS + 1.85, N_COLS + 1.7],
                [bottom_layer, bottom_layer, top_layer, top_layer],
                color="#888", linewidth=0.7, zorder=2)

    ax.text(-1.0, 0, "input", ha="right", va="center", fontsize=9, color="#444")

    span_y = -0.7
    ax.add_patch(FancyArrowPatch(
        (0, span_y), (N_COLS - 1, span_y),
        arrowstyle="<->", mutation_scale=10,
        color=SPAN_COLOR, linewidth=1.0, zorder=2,
    ))
    ax.text((N_COLS - 1) / 2, span_y - 0.35,
            f"RF $= {RF}$ frames",
            ha="center", va="top", fontsize=10, color=SPAN_COLOR)

    ax.set_xlim(-3.0, N_COLS + 4.5)
    ax.set_ylim(-1.4, N_LAYERS + 0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()

    return save(fig, out_name)


def main() -> None:
    setup_style()
    out = render(out_name="tcn_rf_blocks")
    print(f"Wrote {out} (+ .svg)")


if __name__ == "__main__":
    main()
