# scripts/visualizations/dataset_pie.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Pie-of-pie chart for dataset class & subclass distribution (full tier).

Renders the color version used in the thesis main text:
  * dataset_pie_color.{svg,pdf}

Reads totals from results/dataset_stats.txt are baked in below (full tier
totals as of 2026-05-07; rerun upstream stats script and update if regenerated).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.transforms import offset_copy

from _common import REPO_ROOT, setup_style

OUT_DIR = REPO_ROOT / "docs" / "figures" / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_svg(fig, name: str) -> Path:
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf

CLASS_TOTAL = {
    "speech":   2401.09,
    "music":    2362.67,
    "background": 1200.02,
}

SPEECH_SUB = {
    "clean":        960.11,
    "dirty":        480.11,
    "noisy":        480.17,
    "multispeaker": 180.17,
    "som":          180.37,
    "msom":         120.16,
}

MUSIC_SUB = {
    "electronic":   343.35,
    "folk":         343.35,
    "hip-hop":      342.87,
    "instrumental": 342.86,
    "pop":          342.87,
    "rock":         342.86,
    "acapella":     304.51,
}

SPEECH_COLOR   = "#42A5F5"
MUSIC_COLOR    = "#FF5252"
INACTIVE_COLOR = "#455A64"


def shade_ramp(base_hex: str, n: int, lo: float = 0.55, hi: float = 1.0):
    base = np.array([int(base_hex[i:i+2], 16) / 255 for i in (1, 3, 5)])
    return [tuple(base * t + (1 - t) * 1.0) for t in np.linspace(hi, lo, n)]


def draw_subpie(ax_pie, ax_legend, sub: dict, base_color: str, title: str):
    """Draw a subpie in ax_pie, and a corresponding legend in ax_legend.
    Slices are sorted largest → smallest and laid out starting at 12 o'clock,
    going clockwise."""
    items = sorted(sub.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    n = len(labels)

    colors = shade_ramp(base_color, n)
    edge, edge_w = "white", 0.5

    radius = 0.85
    wedges, _ = ax_pie.pie(
        values, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(linewidth=edge_w, edgecolor=edge), radius=radius,
    )
    ax_pie.add_artist(plt.Circle(
        (0, 0), 0.008, facecolor="white", edgecolor="none", zorder=10,
    ))

    total = sum(values)
    handles, leg_labels = [], []
    for w, name, val, c in zip(wedges, labels, values, colors):
        pct = 100 * val / total
        handles.append(Patch(facecolor=c, edgecolor=edge, linewidth=edge_w))
        leg_labels.append(f"{name}  {pct:.1f}%  ({val:.0f} min)")

    ax_legend.legend(
        handles, leg_labels, loc="center", frameon=False, fontsize=14,
        handlelength=1.3, handleheight=1.0, handletextpad=0.5, labelspacing=0.5,
    )
    ax_legend.set_axis_off()

    ax_pie.set_title(title, fontsize=16, pad=4)
    ax_pie.set_aspect("equal")
    ax_pie.set_xlim(-1.0, 1.0)
    ax_pie.set_ylim(-1.0, 1.0)
    ax_pie.set_axis_off()
    return wedges


def render(out_name: str) -> Path:
    fig: Figure = Figure(figsize=(11.0, 6.0))
    FigureCanvasAgg(fig)

    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 0.32],
        wspace=0.05, hspace=0.0,
    )
    ax_speech    = fig.add_subplot(gs[0, 0])
    ax_main      = fig.add_subplot(gs[0, 1])
    ax_music     = fig.add_subplot(gs[0, 2])
    ax_sp_legend = fig.add_subplot(gs[1, 0])
    ax_mu_legend = fig.add_subplot(gs[1, 2])

    total_min    = sum(CLASS_TOTAL.values())
    inactive_ang = 360 * CLASS_TOTAL["background"] / total_min
    startangle   = 90 + inactive_ang / 2

    main_values = [CLASS_TOTAL["speech"], CLASS_TOTAL["music"], CLASS_TOTAL["background"]]
    main_labels = ["speech", "music", "background"]
    main_colors = [SPEECH_COLOR, MUSIC_COLOR, INACTIVE_COLOR]
    edge_color, edge_w = "white", 1.5

    main_wedges, _ = ax_main.pie(
        main_values, colors=main_colors, startangle=startangle,
        counterclock=True, radius=1.0,
        wedgeprops=dict(linewidth=edge_w, edgecolor=edge_color),
    )

    for w, name, val, fill_spec in zip(main_wedges, main_labels, main_values, main_colors):
        ang = (w.theta2 + w.theta1) / 2.0
        x = 0.55 * np.cos(np.deg2rad(ang))
        y = 0.55 * np.sin(np.deg2rad(ang))
        pct = 100 * val / total_min
        ax_main.text(
            x, y, f"{name}\n{val/60:.1f} h\n{pct:.1f}%",
            ha="center", va="center", fontsize=12, color="white", weight="bold",
        )

    ax_main.set_title("Top-level class split", fontsize=16, pad=4)
    ax_main.set_aspect("equal")
    ax_main.set_xlim(-1.15, 1.15)
    ax_main.set_ylim(-1.15, 1.15)
    ax_main.set_axis_off()

    draw_subpie(ax_speech, ax_sp_legend, SPEECH_SUB, SPEECH_COLOR, "Speech subclasses")
    draw_subpie(ax_music,  ax_mu_legend, MUSIC_SUB,  MUSIC_COLOR,  "Music subclasses")

    speech_w, music_w = main_wedges[0], main_wedges[1]
    line_color = "0.55"
    sub_r = 0.85

    def edge_pt(wedge, theta_deg):
        return (wedge.r * np.cos(np.deg2rad(theta_deg)),
                wedge.r * np.sin(np.deg2rad(theta_deg)))

    GAP_MAIN_PT     = 7
    GAP_SUB_TOP_PT  = 17
    GAP_SUB_BOT_PT  = 12
    SUB_Y_OFFSET_TOP_PT  = 2
    SUB_Y_OFFSET_BOT_PT  = 1
    MAIN_Y_OFFSET_BOT_PT = 1

    def y_offset_trans(ax, dy_pt):
        """ax.transData with a vertical offset (in points) applied."""
        return offset_copy(ax.transData, fig=fig, x=0,
                           y=dy_pt, units="points")

    sp_top_main = edge_pt(speech_w, speech_w.theta1)
    sp_bot_main = edge_pt(speech_w, speech_w.theta2)
    sp_top_sub  = (0.0,  sub_r)
    sp_bot_sub  = (0.0, -sub_r)
    for a, b, gap_b, dy_main, dy_sub in [
        (sp_top_main, sp_top_sub, GAP_SUB_TOP_PT, 0,                    +SUB_Y_OFFSET_TOP_PT),
        (sp_bot_main, sp_bot_sub, GAP_SUB_BOT_PT, -MAIN_Y_OFFSET_BOT_PT, -SUB_Y_OFFSET_BOT_PT),
    ]:
        fig.add_artist(ConnectionPatch(
            xyA=a, coordsA=y_offset_trans(ax_main, dy_main),
            xyB=b, coordsB=y_offset_trans(ax_speech, dy_sub),
            color=line_color, linewidth=0.8, linestyle="--",
            shrinkA=GAP_MAIN_PT, shrinkB=gap_b,
        ))

    mu_top_main = edge_pt(music_w, music_w.theta2)
    mu_bot_main = edge_pt(music_w, music_w.theta1)
    mu_top_sub  = (0.0,  sub_r)
    mu_bot_sub  = (0.0, -sub_r)
    for a, b, gap_b, dy_main, dy_sub in [
        (mu_top_main, mu_top_sub, GAP_SUB_TOP_PT, 0,                    +SUB_Y_OFFSET_TOP_PT),
        (mu_bot_main, mu_bot_sub, GAP_SUB_BOT_PT, -MAIN_Y_OFFSET_BOT_PT, -SUB_Y_OFFSET_BOT_PT),
    ]:
        fig.add_artist(ConnectionPatch(
            xyA=a, coordsA=y_offset_trans(ax_main, dy_main),
            xyB=b, coordsB=y_offset_trans(ax_music, dy_sub),
            color=line_color, linewidth=0.8, linestyle="--",
            shrinkA=GAP_MAIN_PT, shrinkB=gap_b,
        ))

    fig.suptitle(
        f"Dataset composition (full tier — {total_min/60:.1f} h, {total_min:.0f} min)",
        fontsize=18, weight="bold", y=0.99,
    )
    fig.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)
    return save_svg(fig, out_name)


def main():
    setup_style()
    color_pdf = render(out_name="dataset_pie_color")
    print(f"Wrote {color_pdf} (+ .svg)")


if __name__ == "__main__":
    main()
