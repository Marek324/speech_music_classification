# scripts/visualizations/combined.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Combined-winners 2^3 factorial visualization.

Replaces / complements tab:combined_results in ch6 §sec:exp_combined.

Plot rationale:
  Same forest-plot layout as the rest of ch6: baseline pinned at top, then
  factorial corners ordered by macro F1 descending. Color codes the CI test
  outcome — F+P+A is the chapter's only strict winner. The y-axis labels
  themselves encode which axes are enabled, so no extra indicator dots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "tcn_combined" / "results"

VARIANTS = [
    ("baseline",            "Baseline",  33),
    ("delta2_conv1d_large", "F+P+A",    480),
    ("delta2_large",        "F+A",      132),
    ("delta2_conv1d",       "F+P",      383),
    ("conv1d_large",        "P+A",      166),
]


def main():
    setup_style()
    rows = []
    for name, label, params_k in VARIANTS:
        data = parse_eval(RESULTS_DIR / f"tcn_{name}.eval")
        ci = data.get("macro_f1_ci")
        rows.append({
            "name": name, "label": label, "params_k": params_k,
            "f1": data["macro_f1"],
            "ci_lo": ci[0] if ci else None,
            "ci_hi": ci[1] if ci else None,
        })

    baseline = next(r for r in rows if r["name"] == "baseline")
    base_lo, base_hi = baseline["ci_lo"], baseline["ci_hi"]

    for r in rows:
        if r["name"] == "baseline":
            r["color"] = COLORS["baseline"]
        elif r["f1"] > base_hi:
            r["color"] = COLORS["winner"]
        elif r["f1"] < base_lo:
            r["color"] = COLORS["bad"]
        else:
            r["color"] = COLORS["neutral"]

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    y = list(range(len(rows)))[::-1]
    f1 = [r["f1"] for r in rows]
    err_lo = [r["f1"] - r["ci_lo"] if r["ci_lo"] else 0.0 for r in rows]
    err_hi = [r["ci_hi"] - r["f1"] if r["ci_hi"] else 0.0 for r in rows]

    ax.barh(y, f1, color=[r["color"] for r in rows],
            alpha=0.85, edgecolor="black", lw=0.4)
    ax.errorbar(f1, y, xerr=[err_lo, err_hi], fmt="none",
                ecolor="black", capsize=3, elinewidth=0.7)

    ax.axvspan(base_lo, base_hi, color="gray", alpha=0.15, zorder=0)
    ax.axvline(baseline["f1"], color="gray", lw=0.6, ls="--", alpha=0.6)

    for yi, r in zip(y, rows):
        x = (r["ci_hi"] if r["ci_hi"] else r["f1"]) + 0.0015
        text = f"{r['f1']:.4f}    ({r['params_k']}K)"
        ax.text(x, yi, text, va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Macro F1")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0.95, 1.00)

    out = save_svg(fig, "combined")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
