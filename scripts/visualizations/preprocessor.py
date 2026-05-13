"""Learned-preprocessor scan — horizontal forest plot.

Replaces / complements tab:preprocessor_results in ch6 §subsec:exp_preprocessor.

Plot rationale:
  Two preprocessor variants (1-D and 2-D conv) plus baseline. Same layout as
  fig:architecture_results: bars sorted with baseline at top, gray band marks
  the baseline CI, color codes the CI test outcome, and added param count is
  appended after the F1 readout so the reader can compare cost row by row.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "nn_preprocessor" / "results"

# Added-parameter counts (kparams), computed from src/nn/preprocessors.py with
# n_features=80 (the 80-mel baseline frontend used in this scan):
#   conv1d: two stacked CausalConv1d(80, 80, k=3) + BatchNorm1d(80)
#           → 2 * (80*80*3 + 80) + 2 * (2*80) ≈ 38.9K
#   conv2d: _CausalConv2d(1, 32, (3,3)) + BatchNorm2d(32)
#         + _CausalConv2d(32, 1, (3,3)) + BatchNorm2d(1)
#           → (1*32*9 + 32) + (2*32) + (32*1*9 + 1) + (2*1) ≈ 0.7K
VARIANTS = [
    ("baseline", "Baseline",            33.0, True),
    ("conv1d",   "Channel-mixing 1-D",  38.9, False),
    ("conv2d",   "Spectro-temporal 2-D", 0.7, False),
]


def _format_params(params_k: float, is_absolute: bool) -> str:
    if is_absolute:
        return f"({params_k:.0f}K)"
    if params_k < 1.0:
        return f"(+{params_k * 1000:.0f})"
    return f"(+{params_k:.1f}K)"


def main():
    setup_style()
    rows = []
    for name, label, params_k, is_abs in VARIANTS:
        data = parse_eval(RESULTS_DIR / f"tcn_{name}.eval")
        ci = data.get("macro_f1_ci")
        rows.append({
            "name": name, "label": label, "params_k": params_k, "is_abs": is_abs,
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

    fig, ax = plt.subplots(figsize=(5.6, 2.4))
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
        if r["name"] == "baseline":
            text = f"{r['f1']:.4f}"
        else:
            text = f"{r['f1']:.4f}    {_format_params(r['params_k'], r['is_abs'])}"
        ax.text(x, yi, text, va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Macro F1")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0.95, 1.00)

    out = save_svg(fig, "preprocessor")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
