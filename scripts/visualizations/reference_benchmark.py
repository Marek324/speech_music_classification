"""Reference benchmark across the four models from §6.1.

Replaces tab:reference_benchmark.

Plot rationale:
  The opening result of the chapter is the macro F1 spread between
  the causal TCN and the three classical baselines, plus how that
  spread distributes across the three classes (Speech, Music,
  Background). A grouped per-class bar chart with macro F1 marked
  separately makes both visible at a glance.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from _common import CLASS_COLORS, MODEL_SHORT, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "results"

MODEL_ORDER = ["tcn", "svm", "decision_tree", "gmm"]


def main():
    setup_style()
    macro = {}
    per_class = {}
    for m in MODEL_ORDER:
        data = parse_eval(RESULTS_DIR / f"{m}.eval")
        macro[m] = data["macro_f1"]
        per_class[m] = data["per_class"]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    n_models = len(MODEL_ORDER)
    width = 0.26
    x = np.arange(n_models)

    classes = ["speech", "music", "background"]
    class_labels = ["Speech", "Music", "Background"]
    for i, (cls, lbl) in enumerate(zip(classes, class_labels)):
        offsets = (i - 1) * width
        vals = [per_class[m][cls]["f1"] for m in MODEL_ORDER]
        ax.bar(
            x + offsets, vals, width,
            color=CLASS_COLORS[cls], alpha=0.85, edgecolor="black", lw=0.4,
            label=lbl,
        )

    # Macro F1 marker at the macro value; label placed above the tallest
    # per-class bar in the same group so it never overlaps a bar.
    for i, m in enumerate(MODEL_ORDER):
        ax.plot(
            i, macro[m], marker="D", markersize=7,
            markerfacecolor="#FFFFFF", markeredgecolor="#000000",
            markeredgewidth=1.0, linestyle="None", zorder=4,
        )
        top_bar = max(per_class[m][cls]["f1"] for cls in classes)
        ax.annotate(
            f"{macro[m]:.3f}",
            xy=(i, top_bar),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=7, color="black",
            annotation_clip=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel("F1")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, ncol=3)

    out = save_svg(fig, "reference_benchmark")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
