# scripts/visualizations/stacks.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Receptive-field / stacks scan on the small-TCN base.

Replaces tab:stacks_results in ch6 §sec:exp_small_tcn_stacks.

Plot rationale:
  Three points on a monotone curve. The argument is that F1 plateaus
  early. The deployed lightweight config (1 stack = TCN-S) is annotated;
  the 3-stack baseline of the small-TCN scan is annotated for context.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "small_tcn_stacks" / "results"

VARIANTS = [
    ("stacks_1", 1, 121, 2.81, "TCN-S"),
    ("stacks_2", 2, 241, 5.60, None),
    ("stacks_3", 3, 361, 8.38, "small-TCN base"),
]


def main():
    setup_style()
    points = []
    for name, stacks, rf_frames, rf_s, label in VARIANTS:
        eval_path = RESULTS_DIR / f"tcn_{name}.eval"
        data = parse_eval(eval_path)
        ci = data.get("macro_f1_ci")
        points.append({
            "stacks": stacks, "rf_frames": rf_frames, "rf_s": rf_s,
            "f1": data["macro_f1"],
            "ci_lo": ci[0] if ci else None,
            "ci_hi": ci[1] if ci else None,
            "label": label,
        })

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    xs = [p["rf_s"] for p in points]
    ys = [p["f1"] for p in points]
    err_lo = [p["f1"] - p["ci_lo"] if p["ci_lo"] else 0.0 for p in points]
    err_hi = [p["ci_hi"] - p["f1"] if p["ci_hi"] else 0.0 for p in points]

    ax.errorbar(
        xs, ys, yerr=[err_lo, err_hi],
        fmt="-", color=COLORS["good"], lw=1.5, capsize=3, alpha=0.6,
        ecolor="black", elinewidth=0.7,
    )
    for p in points:
        is_named = p["label"] is not None
        color = COLORS["winner"] if is_named else COLORS["good"]
        ax.plot(
            p["rf_s"], p["f1"], marker="o",
            markersize=8 if is_named else 6,
            color=color, zorder=3,
        )
        ax.annotate(
            f"{p['f1']:.4f}\n({p['stacks']} stack{'s' if p['stacks'] != 1 else ''})",
            xy=(p["rf_s"], p["f1"]),
            xytext=(0, -22), textcoords="offset points",
            ha="center", fontsize=7,
            color=color,
        )
        if is_named:
            ax.annotate(
                p["label"],
                xy=(p["rf_s"], p["f1"]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8, color=COLORS["winner"], weight="bold",
            )

    ax.axvline(2.97, ls="--", color="gray", lw=0.8, alpha=0.6)
    ax.text(
        2.97, 0.965, "training chunk length",
        rotation=90, ha="right", va="top",
        fontsize=7, color="gray", alpha=0.8,
    )

    ax.set_xlabel("Receptive field (s)")
    ax.set_ylabel("Macro F1")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylim(0.964, 0.982)

    out = save_svg(fig, "stacks")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
