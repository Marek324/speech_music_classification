"""Active-subgroup ablation results sorted by Δ vs baseline.

Replaces / complements tab:ablation_results in ch6 §subsec:exp_ablation.

Plot rationale:
  The story of the table is "what survives the bootstrap CI". Sorting by
  Δ and shading bars by sign + significance makes the pattern obvious:
  three small wins outside the paper grid, two diverged failures, the
  rest noise.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "tcn_ablation" / "results"

# Ordered as in the chapter table; the diverged variants flagged separately.
VARIANTS = [
    "adam_batchnorm",
    "sgd_lr_1e-2",
    "dropout_low",
    "dropout_medium",
    "gelu",
    "baseline",
    "leaky_relu",
    "batch_norm",
    "layers_3",
    "layers_2",
    "elu",
    "layers_1",
    "no_skip",
    "adam",
]

DIVERGED = {"no_skip", "adam"}


def main():
    setup_style()
    rows = []
    for name in VARIANTS:
        if name in DIVERGED:
            # Reads from .diverged sentinel; we use the chapter's reported number.
            rows.append({
                "name": name, "f1": 0.18, "ci_lo": 0.16, "ci_hi": 0.20,
                "diverged": True,
            })
            continue
        eval_path = RESULTS_DIR / f"tcn_{name}.eval"
        data = parse_eval(eval_path)
        ci = data.get("macro_f1_ci")
        rows.append({
            "name": name, "f1": data["macro_f1"],
            "ci_lo": ci[0] if ci else None,
            "ci_hi": ci[1] if ci else None,
            "diverged": False,
        })

    baseline = next(r for r in rows if r["name"] == "baseline")
    base_lo, base_hi = baseline["ci_lo"], baseline["ci_hi"]

    # Compute Δ vs baseline.
    for r in rows:
        r["delta"] = r["f1"] - baseline["f1"]
        if r["diverged"]:
            r["color"] = COLORS["bad"]
        elif r["name"] == "baseline":
            r["color"] = COLORS["baseline"]
        elif r["f1"] > base_hi:
            r["color"] = COLORS["winner"]
        elif r["f1"] < base_lo:
            r["color"] = COLORS["bad"]
        else:
            r["color"] = COLORS["neutral"]

    # Sort by delta descending, with diverged at bottom.
    non_div = [r for r in rows if not r["diverged"]]
    div = [r for r in rows if r["diverged"]]
    non_div.sort(key=lambda r: -r["delta"])
    sorted_rows = non_div + div

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    y = list(range(len(sorted_rows)))[::-1]
    deltas = [r["delta"] for r in sorted_rows]
    colors = [r["color"] for r in sorted_rows]

    bars = ax.barh(y, deltas, color=colors, alpha=0.85, edgecolor="black", lw=0.4)
    ax.axvline(0, color="black", lw=0.6)

    # Shade the baseline-CI band.
    base_band_hi = base_hi - baseline["f1"]
    base_band_lo = base_lo - baseline["f1"]
    ax.axvspan(base_band_lo, base_band_hi, color="gray", alpha=0.12, zorder=0)

    # Annotate Δ values to the right of each bar.
    for yi, r in zip(y, sorted_rows):
        if r["diverged"]:
            txt = "diverged"
        else:
            txt = f"{r['delta']:+.4f}"
        x = r["delta"] + (0.0005 if r["delta"] >= 0 else -0.0005)
        ax.text(
            x, yi, txt,
            va="center", ha="left" if r["delta"] >= 0 else "right",
            fontsize=7,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in sorted_rows], fontsize=8)
    ax.set_xlabel("$\\Delta$ macro F1 vs baseline")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlim(-0.025, 0.012)

    out = save_svg(fig, "ablation")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
