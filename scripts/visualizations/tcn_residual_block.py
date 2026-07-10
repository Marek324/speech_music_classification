# scripts/visualizations/tcn_residual_block.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Figure 3.4: TCN residual block internals.

Two causal dilated convolutions stacked with weight normalization, ReLU, and
dropout after each, plus a skip connection (with a 1x1 projection when input
and output channel counts differ). Drawn bottom-up: input enters at the bottom
and the residual sum produces the output at the top.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from _common import REPO_ROOT, setup_style

OUT_DIR = REPO_ROOT / "thesis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONV_COLOR = "#E3F2FD"
NORM_COLOR = "#FFF8E1"
ACT_COLOR = "#E8F5E9"
DROP_COLOR = "#FCE4EC"
SKIP_COLOR = "#ECEFF1"


def save(fig: Figure, name: str) -> Path:
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


def render(out_name: str) -> Path:
    r"""
    Sized so a moderate `\includegraphics[width=0.7\textwidth]` (≈4.4").
    Gives on-page text close to body-text size without consuming a full
    page. Aspect roughly 1.55 keeps the figure tall enough for the stack
    of nine boxes but leaves head- and footroom on the page.
    """
    fig: Figure = Figure(figsize=(5.5, 8.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    main_x = 0.55
    box_w = 0.30
    box_h = 0.045

    def box(x, y, text, color, w=box_w, h=box_h, fontsize=14, ec="#555"):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.004",
            facecolor=color, edgecolor=ec, linewidth=0.8, zorder=2,
        ))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)
        return (x, y, w, h)

    def arrow(p1, p2, color="#444", lw=1.1, shrinkA=2, shrinkB=2):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=11,
            color=color, linewidth=lw, zorder=1, shrinkA=shrinkA, shrinkB=shrinkB,
        ))

    step = 0.080
    group_gap = 0.045
    y_input  = 0.030
    y_conv1  = 0.125
    y_wn1    = y_conv1 + step
    y_act1   = y_wn1   + step
    y_drop1  = y_act1  + step
    y_conv2  = y_drop1 + step + group_gap
    y_wn2    = y_conv2 + step
    y_act2   = y_wn2   + step
    y_drop2  = y_act2  + step
    y_add     = y_drop2 + 0.085
    y_act_out = y_add    + 0.085
    y_output  = y_act_out + 0.075

    ax.text(main_x, y_input, "input", ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.text(main_x, y_output, "output", ha="center", va="center",
            fontsize=14, fontweight="bold")

    box(main_x, y_conv1, "Conv $(K, d)$", CONV_COLOR)
    box(main_x, y_wn1,   "WeightNorm",   NORM_COLOR)
    box(main_x, y_act1,  "$\\sigma$",    ACT_COLOR, fontsize=16)
    box(main_x, y_drop1, "Dropout",      DROP_COLOR)
    box(main_x, y_conv2, "Conv $(K, d)$", CONV_COLOR)
    box(main_x, y_wn2,   "WeightNorm",   NORM_COLOR)
    box(main_x, y_act2,  "$\\sigma$",    ACT_COLOR, fontsize=16)
    box(main_x, y_drop2, "Dropout",      DROP_COLOR)
    box(main_x, y_act_out, "$\\sigma$",  ACT_COLOR, fontsize=16)

    fig_w, fig_h = fig.get_size_inches()
    add_w = 0.06
    add_h = add_w * (fig_w / fig_h)
    ax.add_patch(Ellipse(
        (main_x, y_add), add_w, add_h,
        facecolor="white", edgecolor="#444", linewidth=1.0, zorder=3,
    ))
    ax.text(main_x, y_add, "$+$", ha="center", va="center",
            fontsize=14, fontweight="bold", zorder=4)

    arrow((main_x, y_input + 0.020), (main_x, y_conv1 - box_h / 2), shrinkA=0)
    arrow((main_x, y_conv1 + box_h / 2), (main_x, y_wn1 - box_h / 2))
    arrow((main_x, y_wn1   + box_h / 2), (main_x, y_act1 - box_h / 2))
    arrow((main_x, y_act1  + box_h / 2), (main_x, y_drop1 - box_h / 2))
    arrow((main_x, y_drop1 + box_h / 2), (main_x, y_conv2 - box_h / 2))
    arrow((main_x, y_conv2 + box_h / 2), (main_x, y_wn2 - box_h / 2))
    arrow((main_x, y_wn2   + box_h / 2), (main_x, y_act2 - box_h / 2))
    arrow((main_x, y_act2  + box_h / 2), (main_x, y_drop2 - box_h / 2))
    arrow((main_x, y_drop2 + box_h / 2), (main_x, y_add - add_h / 2), shrinkB=0)
    arrow((main_x, y_add   + add_h / 2), (main_x, y_act_out - box_h / 2), shrinkA=0)
    arrow((main_x, y_act_out + box_h / 2), (main_x, y_output - 0.020))

    skip_x = 0.84
    skip_branch_y = ((y_input + 0.020) + (y_conv1 - box_h / 2)) / 2

    proj_y = (skip_branch_y + y_add) / 2
    n_skip = box(skip_x, proj_y, "$1{\\times}1$ proj",
                 SKIP_COLOR, w=0.20, h=0.045, fontsize=12)
    n_skip_top = n_skip[1] + n_skip[3] / 2
    n_skip_bot = n_skip[1] - n_skip[3] / 2

    ax.plot([main_x, skip_x], [skip_branch_y, skip_branch_y],
            color="#444", linewidth=1.0, zorder=1)
    ax.plot([skip_x, skip_x], [skip_branch_y, n_skip_bot],
            color="#444", linewidth=1.0, zorder=1)
    ax.plot([skip_x, skip_x], [n_skip_top, y_add],
            color="#444", linewidth=1.0, zorder=1)
    arrow((skip_x, y_add), (main_x + add_w / 2, y_add), shrinkA=0, shrinkB=0)

    ax.text(skip_x + 0.030, (n_skip_top + y_add) / 2,
            "skip", ha="left", va="center", fontsize=12, color="#666",
            style="italic", rotation=90)

    leg_x = 0.00
    swatch_w, swatch_h = 0.040, 0.030
    label_fs = 10
    line_step = 0.024
    entry_gap = 0.040

    entries = [
        (CONV_COLOR, ["$\\mathrm{Conv}(K, d)$: causal", "dilated convolution"]),
        (ACT_COLOR,  ["$\\sigma$: non-linearity"]),
    ]

    entry_heights = [
        max(swatch_h, swatch_h / 2 + (len(lines) - 1) * line_step + line_step / 2)
        for _, lines in entries
    ]
    total_h = sum(entry_heights) + entry_gap * (len(entries) - 1)

    text_x = leg_x + swatch_w + 0.014
    cursor = 0.5 + total_h / 2
    for (color, lines), entry_h in zip(entries, entry_heights):
        ax.add_patch(Rectangle(
            (leg_x, cursor - swatch_h), swatch_w, swatch_h,
            facecolor=color, edgecolor="#555", linewidth=0.8, zorder=2,
        ))
        first_line_y = cursor - swatch_h / 2
        for i, line in enumerate(lines):
            ax.text(text_x, first_line_y - i * line_step, line,
                    ha="left", va="center", fontsize=label_fs, color="#333")
        cursor -= entry_h + entry_gap

    return save(fig, out_name)


def main() -> None:
    setup_style()
    out = render(out_name="tcn_residual_block")
    print(f"Wrote {out} (+ .svg)")


if __name__ == "__main__":
    main()
