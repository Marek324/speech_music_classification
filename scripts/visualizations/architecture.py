"""Backbone architecture scan at two parameter budgets.

Replaces / complements tab:architecture_results in ch6 §subsec:exp_architecture.

Plot rationale:
  Same forest-plot layout as fig:frontend_results / fig:hybrid_results: bars
  sorted with baseline at top, then ascending by parameter count, gray band
  marks the baseline CI, color codes the CI test outcome. Param count is
  appended after the F1 readout so the parameter-vs-accuracy trade-off is
  legible row by row.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "nn_architecture" / "results"

# Sorted with baseline pinned at top, then ascending by total params.
# Param counts are sourced from tab:architecture_results in ch6.
# transformer_large (0.6942 macro F1) is omitted: its bar dwarfs the chart
# scale and its collapse is fully captured by transformer_small. The full
# numbers are kept in the table and discussed in the prose.
VARIANTS = [
    ("baseline",          "TCN ($F{=}16$)",          33),
    ("lstm_small",        "LSTM ($F{=}32$)",         36),
    ("gru_small",         "GRU ($F{=}32$)",          28),
    ("transformer_small", "Transformer ($F{=}24$)",  31),
    ("tcn_large",         "TCN ($F{=}32$)",          127),
    ("gru_large",         "GRU ($F{=}72$)",          132),
    ("lstm_large",        "LSTM ($F{=}64$)",         138),
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

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    # Detect boundary between small (~30K) and large (~130K) budgets by param-gap.
    GAP_SIZE = 0.7
    n = len(rows)
    boundary_idx = next(
        (i for i in range(1, n) if rows[i]["params_k"] - rows[i - 1]["params_k"] > 50),
        n,
    )
    # Lift the small-budget group up by GAP_SIZE so a visible gap opens at the boundary.
    y = [(n - 1 - i) + (GAP_SIZE if i < boundary_idx else 0.0) for i in range(n)]

    f1 = [r["f1"] for r in rows]
    err_lo = [r["f1"] - r["ci_lo"] if r["ci_lo"] else 0.0 for r in rows]
    err_hi = [r["ci_hi"] - r["f1"] if r["ci_hi"] else 0.0 for r in rows]

    ax.barh(y, f1, color=[r["color"] for r in rows],
            alpha=0.85, edgecolor="black", lw=0.4)
    ax.errorbar(f1, y, xerr=[err_lo, err_hi], fmt="none",
                ecolor="black", capsize=3, elinewidth=0.7)

    ax.axvspan(base_lo, base_hi, color="gray", alpha=0.15, zorder=0)
    ax.axvline(baseline["f1"], color="gray", lw=0.6, ls="--", alpha=0.6)

    if 0 < boundary_idx < n:
        divider_y = (y[boundary_idx - 1] + y[boundary_idx]) / 2
        ax.axhline(divider_y, color="gray", lw=0.5, ls="--", alpha=0.5)

    for yi, r in zip(y, rows):
        x = (r["ci_hi"] if r["ci_hi"] else r["f1"]) + 0.005
        text = f"{r['f1']:.4f}    ({r['params_k']}K)"
        ax.text(x, yi, text, va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Macro F1")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0.85, 1.00)

    out = save_svg(fig, "architecture")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
