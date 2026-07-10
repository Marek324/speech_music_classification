# scripts/visualizations/tcn_backbone.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Figure 3.4: TCN backbone overview.

Schematic: 2 stacks of 4 residual blocks each (dilations 1, 2, 4, 8 within
each stack), plus 1x1 input and output projections. The deployed model uses
3 stacks; the figure shows 2 for clarity.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _common import REPO_ROOT, setup_style

OUT_DIR = REPO_ROOT / "thesis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJ_COLOR = "#FFE0B2"
BLOCK_COLOR = "#E3F2FD"
STACK_TINTS = ["#F2F2F2", "#E4E4E4"]


def save(fig: Figure, name: str) -> Path:
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


def render(out_name: str) -> Path:
    fig: Figure = Figure(figsize=(12.0, 2.7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    y_main = 0.45

    def box(x, y, w, h, text, color, fontsize=10, ec="#555"):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.004",
            facecolor=color, edgecolor=ec, linewidth=0.8, zorder=2,
        ))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)
        return (x, y, w, h)

    def arrow(p1, p2, color="#555", lw=0.9):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=10,
            color=color, linewidth=lw, zorder=1, shrinkA=2, shrinkB=2,
        ))

    ax.text(0.045, y_main, "log-mel\n(mel $\\times$ time)",
            ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.955, y_main, "logits\n(3 $\\times$ time)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    proj_w, proj_h = 0.075, 0.22
    n_proj_in = box(0.180, y_main, proj_w, proj_h,
                    "$1{\\times}1$\nconv\n(mel $\\to F$)", PROJ_COLOR, fontsize=8.5)
    n_proj_out = box(0.820, y_main, proj_w, proj_h,
                     "$1{\\times}1$\nconv\n($F \\to 3$)", PROJ_COLOR, fontsize=8.5)

    stack_starts = [0.250, 0.520]
    stack_width = 0.230
    block_w = 0.05
    block_h = 0.13
    block_spacing = 0.052
    block_x_offset = 0.037

    DILATIONS = [1, 2, 4, 8]

    block_centers = []
    for s_idx, x_start in enumerate(stack_starts):
        ax.add_patch(Rectangle(
            (x_start - 0.005, y_main - 0.22), stack_width + 0.01, 0.44,
            facecolor=STACK_TINTS[s_idx], edgecolor="#CCCCCC", linewidth=0.5, zorder=0,
        ))
        ax.text(x_start + stack_width / 2, y_main + 0.27,
                f"stack {s_idx + 1}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#444")

        block_xs = [x_start + block_x_offset + i * block_spacing for i in range(4)]
        for i, bx in enumerate(block_xs):
            box(bx, y_main, block_w, block_h,
                f"$d{{=}}{DILATIONS[i]}$", BLOCK_COLOR, fontsize=10)
            block_centers.append(bx)

    arrow((n_proj_in[0] + proj_w / 2, y_main),
          (block_centers[0] - block_w / 2, y_main))

    for i in range(len(block_centers) - 1):
        x1 = block_centers[i] + block_w / 2
        x2 = block_centers[i + 1] - block_w / 2
        arrow((x1, y_main), (x2, y_main))

    arrow((block_centers[-1] + block_w / 2, y_main),
          (n_proj_out[0] - proj_w / 2, y_main))

    arrow((0.105, y_main), (n_proj_in[0] - proj_w / 2, y_main))
    arrow((n_proj_out[0] + proj_w / 2, y_main), (0.895, y_main))

    return save(fig, out_name)


def main() -> None:
    setup_style()
    out = render(out_name="tcn_backbone")
    print(f"Wrote {out} (+ .svg)")


if __name__ == "__main__":
    main()
