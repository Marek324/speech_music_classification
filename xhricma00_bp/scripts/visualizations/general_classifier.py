# scripts/visualizations/general_classifier.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Figure 3.1: Generic speech/music classifier pipeline.

Schematic: Audio Input -> Signal Pre-processing -> Feature Extraction ->
Classifier -> Decision. Used at the top of chapter 3 to motivate the
classifier zoo discussed in subsequent sections.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import REPO_ROOT, setup_style

OUT_DIR = REPO_ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEUTRAL_COLOR = "#F2F2F2"
CLASSIFIER_COLOR = "#E3F2FD"
DECISION_COLOR = "#FFE0B2"


def save(fig: Figure, name: str) -> Path:
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


def render(out_name: str) -> Path:
    fig: Figure = Figure(figsize=(11.0, 2.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    y_main = 0.55
    y_subtitle = 0.22

    def box(x, y, w, h, text, color, fontsize=11, ec="#555"):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.004",
            facecolor=color, edgecolor=ec, linewidth=0.9, zorder=2,
        ))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, zorder=3)
        return (x, y, w, h)

    def arrow(p1, p2, color="#555", lw=1.1):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=12,
            color=color, linewidth=lw, zorder=4, shrinkA=2, shrinkB=2,
        ))

    def subtitle(x, text):
        ax.text(x, y_subtitle, text, ha="center", va="center",
                fontsize=9, style="italic", color="#666", zorder=3)

    block_h = 0.36
    centers = [0.10, 0.30, 0.52, 0.74, 0.92]
    widths  = [0.13, 0.17, 0.17, 0.17, 0.13]

    audio = box(centers[0], y_main, widths[0], block_h,
                "Audio Input", NEUTRAL_COLOR)
    wave_n = 110
    wave_xs = np.linspace(centers[0] - widths[0] * 0.38,
                          centers[0] + widths[0] * 0.38, wave_n)
    rng = np.random.default_rng(0)
    env = np.exp(-((np.linspace(-1.5, 1.5, wave_n)) ** 2))
    wave_ys = (
        y_subtitle
        + 0.05 * env * np.sin(np.linspace(0, 8 * np.pi, wave_n))
        + 0.008 * env * rng.standard_normal(wave_n)
    )
    ax.plot(wave_xs, wave_ys, color="#777", linewidth=0.9, zorder=3)

    pre = box(centers[1], y_main, widths[1], block_h,
              "Signal\nPre-processing", NEUTRAL_COLOR)
    subtitle(centers[1], "Buffering, Segmentation, ...")

    feat = box(centers[2], y_main, widths[2], block_h,
               "Feature\nExtraction", NEUTRAL_COLOR)
    subtitle(centers[2], "ZCR, MFCC, ...")

    clf = box(centers[3], y_main, widths[3], block_h,
              "Classifier", CLASSIFIER_COLOR, fontsize=12)
    subtitle(centers[3], "GMM, CNN, ...")

    dec = box(centers[4], y_main, widths[4], block_h,
              "Decision", DECISION_COLOR, fontsize=12)
    subtitle(centers[4], "Speech/Music")

    nodes = [audio, pre, feat, clf, dec]
    for left, right in zip(nodes[:-1], nodes[1:]):
        x1 = left[0] + left[2] / 2
        x2 = right[0] - right[2] / 2
        arrow((x1, y_main), (x2, y_main))

    return save(fig, out_name)


def main() -> None:
    setup_style()
    out = render(out_name="general_classifier")
    print(f"Wrote {out} (+ .svg)")


if __name__ == "__main__":
    main()
