# scripts/visualizations/_common.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Shared utilities for chapter-6 visualizations.

Style:
  * Output is SVG, sized for one-column LaTeX (\\textwidth).
  * Palette matches scripts/visualize_results.py.
  * parse_eval() is reused from scripts/visualize_results.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from visualize_results import parse_eval

OUT_DIR = REPO_ROOT / "thesis" / "figures" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "tcn": "#2196F3",
    "tcn_l": "#E91E63",
    "tcn_s": "#4DB6AC",
    "decision_tree": "#4CAF50",
    "svm": "#FF9800",
    "gmm": "#9C27B0",
    "baseline": "#90A4AE",
    "winner": "#E91E63",
    "good": "#2196F3",
    "neutral": "#90A4AE",
    "bad": "#EF5350",
}

MODEL_SHORT = {
    "tcn_l": "TCN-L",
    "tcn_s": "TCN-S",
    "tcn": "TCN",
    "decision_tree": "DT",
    "svm": "SVM",
    "gmm": "GMM",
}

CLASS_COLORS = {
    "speech":   "#42A5F5",
    "music":    "#FF5252",
    "background": "#455A64",
}


def setup_style():
    """Set rcParams for consistent figures across scripts."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def save_svg(fig, name: str) -> Path:
    """Save figure as SVG and PDF to thesis/figures/experiments/<name>.{svg,pdf}.

    PDF is what pdflatex actually includes; SVG is kept alongside for web/preview.
    """
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


__all__ = [
    "CLASS_COLORS",
    "COLORS",
    "MODEL_SHORT",
    "OUT_DIR",
    "REPO_ROOT",
    "parse_eval",
    "save_svg",
    "setup_style",
]
