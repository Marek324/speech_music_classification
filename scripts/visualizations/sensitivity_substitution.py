# scripts/visualizations/sensitivity_substitution.py
# Marek Hric

"""Component-substitution scan — grouped horizontal forest plot.

Replaces / complements tab:sensitivity_substitution in ch6 §subsec:exp_sensitivity.

Plot rationale:
  Three substitution categories (optimizer, normalization, activation) plus
  baseline. Within-category section headers act as horizontal dividers so the
  reader can tell the groups apart without the LaTeX-style \\multicolumn rows.
  Color codes the CI test outcome; baseline gray, winners pink, losers red,
  neutrals slate — same palette as fig:frontend_results.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "tcn_ablation" / "results"

# Top-to-bottom: baseline, then three categories with their variants.
ROWS = [
    {"type": "variant", "name": "baseline",        "label": "Baseline"},
    {"type": "section", "label": "Optimizer"},
    {"type": "variant", "name": "adam_batchnorm",  "label": "adam + batch-norm"},
    {"type": "section", "label": "Normalization"},
    {"type": "variant", "name": "batch_norm",      "label": "batch-norm"},
    {"type": "section", "label": "Activation"},
    {"type": "variant", "name": "gelu",            "label": "GeLU"},
    {"type": "variant", "name": "leaky_relu",      "label": "LeakyReLU"},
    {"type": "variant", "name": "elu",             "label": "ELU"},
]


def _color_for(row, base_lo, base_hi):
    if row["name"] == "baseline":
        return COLORS["baseline"]
    if row["f1"] > base_hi:
        return COLORS["winner"]
    if row["f1"] < base_lo:
        return COLORS["bad"]
    return COLORS["neutral"]


def main():
    """Render the substitution-scan forest plot to docs/figures/experiments/."""
    setup_style()

    for r in ROWS:
        if r["type"] != "variant":
            continue
        data = parse_eval(RESULTS_DIR / f"tcn_{r['name']}.eval")
        ci = data.get("macro_f1_ci")
        r["f1"] = data["macro_f1"]
        r["ci_lo"] = ci[0] if ci else None
        r["ci_hi"] = ci[1] if ci else None

    baseline = next(r for r in ROWS if r.get("name") == "baseline")
    base_lo, base_hi = baseline["ci_lo"], baseline["ci_hi"]

    for r in ROWS:
        if r["type"] == "variant":
            r["color"] = _color_for(r, base_lo, base_hi)

    n = len(ROWS)
    y_pos = list(range(n))[::-1]  # top row at largest y

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    # Baseline CI band + mean line.
    ax.axvspan(base_lo, base_hi, color="gray", alpha=0.15, zorder=0)
    ax.axvline(baseline["f1"], color="gray", lw=0.6, ls="--", alpha=0.6, zorder=0)

    variant_y, variant_f1, variant_lo, variant_hi, variant_colors = [], [], [], [], []
    for yi, r in zip(y_pos, ROWS):
        if r["type"] == "variant":
            variant_y.append(yi)
            variant_f1.append(r["f1"])
            variant_lo.append(r["f1"] - r["ci_lo"] if r["ci_lo"] else 0.0)
            variant_hi.append(r["ci_hi"] - r["f1"] if r["ci_hi"] else 0.0)
            variant_colors.append(r["color"])

    ax.barh(variant_y, variant_f1, color=variant_colors,
            alpha=0.85, edgecolor="black", lw=0.4)
    ax.errorbar(variant_f1, variant_y, xerr=[variant_lo, variant_hi], fmt="none",
                ecolor="black", capsize=3, elinewidth=0.7)

    # Annotate F1 just past each row's upper-CI whisker.
    for yi, r in zip(y_pos, ROWS):
        if r["type"] != "variant":
            continue
        x = (r["ci_hi"] if r["ci_hi"] else r["f1"]) + 0.0015
        ax.text(x, yi, f"{r['f1']:.4f}",
                va="center", ha="left", fontsize=7)

    # Section dividers — thin axhline aligned with the category label row.
    for yi, r in zip(y_pos, ROWS):
        if r["type"] == "section":
            ax.axhline(yi, color="black", lw=0.4, alpha=0.5, zorder=1)

    # Y-tick labels — section headers bold, no tick mark on those rows.
    ax.set_yticks(y_pos)
    text_objs = ax.set_yticklabels([r["label"] for r in ROWS])
    for text_obj, r in zip(text_objs, ROWS):
        if r["type"] == "section":
            text_obj.set_fontweight("bold")

    section_ys = {y for y, r in zip(y_pos, ROWS) if r["type"] == "section"}
    for tick in ax.yaxis.get_major_ticks():
        if tick.get_loc() in section_ys:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

    ax.set_xlabel("Macro F1")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0.95, 1.00)

    out = save_svg(fig, "sensitivity_substitution")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
