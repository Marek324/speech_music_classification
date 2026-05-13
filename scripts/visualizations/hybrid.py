"""Temporal-head scan on the F+P TCN base.

Replaces / complements tab:hybrid_results in ch5 §subsec:exp_hybrid.

Plot rationale:
  Four heads on top of the F+P base, sorted by macro F1. The chapter
  conclusion is "no head clears the F+P base CI"; the gray band marking
  the F+P-base CI shows that all heads sit inside the band.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "tcn_temporal_head" / "results"

# Sorted with F+P base pinned at top, then heads by added params.
# The F+P base shows its absolute parameter count; heads show params added on top of it.
# `tcn_lstm` is the variant adopted as TCN-L; read its score from the deployed
# checkpoint's eval file (results/tcn_l.eval) so the plot value matches the
# rest of the thesis.
VARIANTS = [
    ("baseline",       "F+P (no head)",     383, True),
    ("tcn_gru_wide",   "GRU $F{=}64$",       16, False),
    ("tcn_gru",        "GRU $F{=}32$",        5, False),
    ("tcn_lstm",       "LSTM $F{=}32$",       6, False),
    ("tcn_attn",       "causal attention",    7, False),
]

LSTM_EVAL_OVERRIDE = REPO_ROOT / "results" / "tcn_l.eval"


def main():
    setup_style()
    rows = []
    for name, label, params_k, is_abs in VARIANTS:
        eval_path = LSTM_EVAL_OVERRIDE if name == "tcn_lstm" else RESULTS_DIR / f"tcn_{name}.eval"
        data = parse_eval(eval_path)
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

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
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
        suffix = f"({r['params_k']}K)" if r["is_abs"] else f"(+{r['params_k']}K)"
        text = f"{r['f1']:.4f}    {suffix}"
        ax.text(x, yi, text, va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Macro F1")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0.95, 1.00)

    out = save_svg(fig, "hybrid")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
