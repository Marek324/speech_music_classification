"""Filter-count Pareto scan on the small-TCN base.

Replaces tab:filter_pareto_results in ch6 §sec:exp_small_tcn_pareto.

Plot rationale:
  Filter count is a continuous parameter, F1 is a continuous outcome,
  the underlying argument is a knee. A line chart with the Pareto knee
  highlighted shows it at a glance; a 4-row table forces the reader to
  mentally interpolate.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _common import COLORS, REPO_ROOT, parse_eval, save_svg, setup_style

RESULTS_DIR = REPO_ROOT / "src" / "exp" / "small_tcn_pareto" / "results"

VARIANTS = [
    ("filters_4", 4),
    ("filters_6", 6),
    ("filters_8", 8),  # deployed
    ("filters_12", 12),
]
DEPLOYED_FILTERS = 8


def main():
    setup_style()
    points = []
    for name, n_filters in VARIANTS:
        eval_path = RESULTS_DIR / f"tcn_{name}.eval"
        data = parse_eval(eval_path)
        ci = data.get("macro_f1_ci")
        points.append({
            "n_filters": n_filters,
            "f1": data["macro_f1"],
            "ci_lo": ci[0] if ci else None,
            "ci_hi": ci[1] if ci else None,
            "deployed": n_filters == DEPLOYED_FILTERS,
        })

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    xs = [p["n_filters"] for p in points]
    ys = [p["f1"] for p in points]
    err_lo = [p["f1"] - p["ci_lo"] if p["ci_lo"] else 0.0 for p in points]
    err_hi = [p["ci_hi"] - p["f1"] if p["ci_hi"] else 0.0 for p in points]

    ax.errorbar(
        xs, ys, yerr=[err_lo, err_hi],
        fmt="-", color=COLORS["good"], lw=1.5, capsize=3, alpha=0.6,
        ecolor="black", elinewidth=0.7,
    )
    # Plot points, deployed in a different color.
    for p in points:
        color = COLORS["winner"] if p["deployed"] else COLORS["good"]
        ax.plot(
            p["n_filters"], p["f1"], marker="o",
            markersize=8 if p["deployed"] else 6,
            color=color, zorder=3,
        )
        # Annotate F1 below the point.
        ax.annotate(
            f"{p['f1']:.4f}",
            xy=(p["n_filters"], p["f1"]),
            xytext=(0, -14), textcoords="offset points",
            ha="center", fontsize=7,
            color=color,
        )

    # Annotate the deployed point.
    deployed = next(p for p in points if p["deployed"])
    ax.annotate(
        "Pareto knee (8 filters)",
        xy=(deployed["n_filters"], deployed["f1"]),
        xytext=(8, 12), textcoords="offset points",
        fontsize=8, color=COLORS["winner"],
        arrowprops=dict(arrowstyle="-", color=COLORS["winner"], lw=0.8),
    )

    ax.set_xticks([p["n_filters"] for p in points])
    ax.set_xlabel("Number of filters $n_{\\mathrm{filters}}$")
    ax.set_ylabel("Macro F1")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylim(0.945, 0.985)

    out = save_svg(fig, "filter_pareto")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
