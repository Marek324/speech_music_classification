# scripts/visualizations/dataset_pipeline.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Three-stage dataset construction pipeline figure for chapter 4.

  Stage 1: Acquire + label  (sources -> labelers -> real subclasses)
  Stage 2: Augment           (real subclasses -> synthetic speech subclasses)
  Stage 3: Split             (per-source per-subclass into train/val/test)
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _common import REPO_ROOT, setup_style

OUT_DIR = REPO_ROOT / "thesis" / "figures" / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPEECH_COLOR = "#42A5F5"
MUSIC_COLOR = "#FF5252"
INACTIVE_COLOR = "#455A64"


def lighten(hex_color: str, factor: float = 0.75) -> tuple[float, float, float]:
    base = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(c + (1 - c) * factor for c in base)


def save(fig: Figure, name: str) -> Path:
    out_svg = OUT_DIR / f"{name}.svg"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


def render(out_name: str) -> Path:
    fig: Figure = Figure(figsize=(13.0, 5.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.18, 1)
    ax.set_axis_off()

    stage_bands = [
        (0.005, 0.49, "1. Acquire + label"),
        (0.49, 0.72, "2. Augment"),
        (0.72, 0.995, "3. Split"),
    ]
    for x0, x1, label in stage_bands:
        ax.text(
            (x0 + x1) / 2, 0.945, label,
            ha="center", va="center", fontsize=14, weight="bold",
        )

    def node(x, y, w, h, text, color, fontsize=9, ec="#555"):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.004",
            facecolor=color, edgecolor=ec, linewidth=0.8, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, zorder=3)
        return (x, y, w, h)

    BOX_PAD = 0.004

    def arrow(p1, p2, color="#666", lw=0.9, style="-|>"):
        ar = FancyArrowPatch(
            p1, p2, color=color, arrowstyle=style,
            mutation_scale=10, linewidth=lw, zorder=4,
            shrinkA=2, shrinkB=2,
        )
        ax.add_patch(ar)

    src_x = 0.06
    src_y = [0.80, 0.68, 0.54, 0.42, 0.28]
    sources = [
        ("LibriSpeech (clean)",   SPEECH_COLOR),
        ("LibriSpeech (other)",   SPEECH_COLOR),
        ("FMA (6 genres)",        MUSIC_COLOR),
        ("Bel Canto",             MUSIC_COLOR),
        ("DEMAND",                INACTIVE_COLOR),
    ]
    src_nodes = []
    for (name, c), y in zip(sources, src_y):
        src_nodes.append(node(src_x, y, 0.15, 0.055, name, lighten(c, 0.82)))

    lab_x = 0.24
    lab_y = [0.74, 0.48, 0.28]
    labelers = [
        ("Silero VAD",                  SPEECH_COLOR),
        ("pyannote + RMS\nthreshold",   MUSIC_COLOR),
        ("trust source",                INACTIVE_COLOR),
    ]
    lab_nodes = []
    for (name, c), y in zip(labelers, lab_y):
        lab_nodes.append(node(lab_x, y, 0.13, 0.07, name, lighten(c, 0.78)))

    real_x = 0.41
    real_y = [0.80, 0.68, 0.54, 0.42, 0.28]
    real_subs = [
        ("clean speech",          SPEECH_COLOR),
        ("dirty speech",          SPEECH_COLOR),
        ("music (6 genres)",      MUSIC_COLOR),
        ("a cappella music",      MUSIC_COLOR),
        ("noise",                 INACTIVE_COLOR),
    ]
    real_nodes = []
    for (name, c), y in zip(real_subs, real_y):
        real_nodes.append(node(real_x, y, 0.12, 0.055, name,
                               lighten(c, 0.7), fontsize=8.5))

    src_to_lab = [0, 0, 1, 1, 2]
    lab_to_real = {0: [0, 1], 1: [2, 3], 2: [4]}
    for i, s in enumerate(src_nodes):
        l = lab_nodes[src_to_lab[i]]
        arrow(
            (s[0] + s[2] / 2 + BOX_PAD, s[1]),
            (l[0] - l[2] / 2 - BOX_PAD, l[1]),
        )
    for li, real_idxs in lab_to_real.items():
        l = lab_nodes[li]
        for ri in real_idxs:
            r = real_nodes[ri]
            arrow(
                (l[0] + l[2] / 2 + BOX_PAD, l[1]),
                (r[0] - r[2] / 2 - BOX_PAD, r[1]),
            )

    aug_x = 0.61
    aug_y = [0.74, 0.61, 0.48, 0.35]
    augs = [
        ("multi-speaker speech\n2× LibriSpeech",                       SPEECH_COLOR),
        ("speech over music (SoM)\nLibriSpeech + FMA",                 SPEECH_COLOR),
        ("multi-speaker\nspeech over music\n2× LibriSpeech + FMA",     SPEECH_COLOR),
        ("noisy speech\nLibriSpeech + DEMAND",                         SPEECH_COLOR),
    ]
    aug_nodes = []
    for (name, c), y in zip(augs, aug_y):
        aug_nodes.append(node(aug_x, y, 0.18, 0.085, name,
                              lighten(c, 0.7), fontsize=8.5))


    bus_x = 0.715
    bus_top = 0.83
    bus_bot = 0.25
    for r in real_nodes:
        ax.plot(
            [r[0] + r[2] / 2 + BOX_PAD, bus_x], [r[1], r[1]],
            color="#bbb", lw=0.6, zorder=1,
        )
    for a in aug_nodes:
        ax.plot(
            [a[0] + a[2] / 2 + BOX_PAD, bus_x], [a[1], a[1]],
            color="#888", lw=0.7, zorder=2,
        )
    ax.plot(
        [bus_x, bus_x], [bus_bot, bus_top],
        color="#888", lw=1.2, zorder=3,
    )

    split_x = 0.81
    split = node(
        split_x, 0.515, 0.075, 0.08,
        "Splitter",
        "#FAFAFA", fontsize=10, ec="#444",
    )

    dataset_x = 0.95
    dataset_w = 0.105
    dataset_h = 0.30
    dataset_y = 0.515
    dataset_left = dataset_x - dataset_w / 2
    dataset_top = dataset_y + dataset_h / 2

    ax.add_patch(FancyBboxPatch(
        (dataset_left, dataset_y - dataset_h / 2), dataset_w, dataset_h,
        boxstyle="round,pad=0.004",
        facecolor="#FAFAFA", edgecolor="#888", linewidth=0.8, zorder=1,
    ))
    ax.text(dataset_x, dataset_top - 0.028, "Dataset",
            ha="center", va="center", fontsize=11, weight="bold", zorder=3)

    sub_w = 0.085
    sub_h = 0.07
    sub_gap = 0.005
    sub_top = dataset_top - 0.06
    parts = [
        ("train (84%)", "#C5E1A5"),
        ("val (8%)",    "#FFE082"),
        ("test (8%)",   "#EF9A9A"),
    ]
    for i, (name, c) in enumerate(parts):
        sy = sub_top - sub_h / 2 - i * (sub_h + sub_gap)
        node(dataset_x, sy, sub_w, sub_h, name, c)

    arrow(
        (bus_x, split[1]),
        (split[0] - split[2] / 2 - BOX_PAD, split[1]),
        color="#444", lw=1.5,
    )
    arrow(
        (split[0] + split[2] / 2 + BOX_PAD, split[1]),
        (dataset_left - BOX_PAD, split[1]),
        color="#444", lw=1.5,
    )
    ax.text(
        bus_x + 0.005, bus_top - 0.025,
        "all 14 subclasses",
        ha="left", va="top", fontsize=8, color="#555", style="italic",
    )

    return save(fig, out_name)


def main() -> None:
    setup_style()
    out = render(out_name="dataset_pipeline")
    print(f"Wrote {out} (+ .svg)")


if __name__ == "__main__":
    main()
