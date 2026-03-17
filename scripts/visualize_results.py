"""Visualize evaluation results for all models."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["tcn", "decision_tree", "svm", "gmm"]
COLORS = {"tcn": "#2196F3", "decision_tree": "#4CAF50", "svm": "#FF9800", "gmm": "#9C27B0"}
MODEL_SHORT = {"tcn": "TCN", "decision_tree": "DT", "svm": "SVM", "gmm": "GMM"}

SPEECH_SUBS = [
    "speech_clean", "speech_corrupted", "speech_multispeaker",
    "speech_multispeaker_over_music", "speech_noisy", "speech_over_music",
]
MUSIC_SUBS = [
    "music_country", "music_electronic", "music_folk", "music_hiphop",
    "music_instrumental", "music_pop", "music_rock", "music_vocal",
]
SPEECH_SHORT = ["clean", "corrupted", "multi", "multi+music", "noisy", "+music"]
MUSIC_SHORT = ["country", "electronic", "folk", "hiphop", "instrumental", "pop", "rock", "vocal"]


def parse_eval(path: Path) -> dict:
    text = path.read_text()

    def macro_f1(section: str) -> float | None:
        m = re.search(r"Macro\s+([\d.]+)", section)
        return float(m.group(1)) if m else None

    eval_section = re.search(r"── 3-class.*?── By subclass", text, re.DOTALL)
    subclass_section = re.search(r"── By subclass.*", text, re.DOTALL)

    time_m = re.search(r"Time/frame\s*:\s*([\d.]+)", text)
    time_per_frame = float(time_m.group(1)) if time_m else None

    per_class = {}
    if eval_section:
        for cls in ["Speech", "Music", "Inactive"]:
            m = re.search(rf"{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", eval_section.group())
            if m:
                per_class[cls.lower()] = {
                    "f1": float(m.group(1)), "p": float(m.group(2)), "r": float(m.group(3)),
                }

    def parse_cm(section_text, classes):
        rows = []
        for cls in classes:
            m = re.search(rf"{cls}\s+\[([\d\s]+)\]", section_text)
            if m:
                rows.append([int(x) for x in m.group(1).split()])
        return np.array(rows, dtype=float) if len(rows) == len(classes) else None

    cm = parse_cm(eval_section.group(), ["Speech", "Music", "Inactive"]) if eval_section else None

    subclasses = {}
    if subclass_section:
        for line in subclass_section.group().splitlines()[1:]:
            m = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                subclasses[m.group(1)] = {
                    "f1": float(m.group(2)), "p": float(m.group(3)), "r": float(m.group(4)),
                }

    return {
        "macro_f1": macro_f1(eval_section.group()) if eval_section else None,
        "per_class": per_class,
        "cm": cm,
        "time_per_frame": time_per_frame,
        "subclasses": subclasses,
    }


def plot_macro_f1(ax, data):
    labels = list(data.keys())
    x = np.arange(len(labels))
    f1_vals = [data[m]["macro_f1"] or 0 for m in labels]
    bars = ax.bar(x, f1_vals, 0.5, color=[COLORS[m] for m in labels], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in labels])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Macro F1 (3-class)")
    ax.set_ylabel("Macro F1")
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7)


def plot_pr_scatter(ax, data):
    # Iso-F1 curves
    r_range = np.linspace(0.5, 1.0, 300)
    for f1_val in [0.75, 0.85, 0.90, 0.95]:
        with np.errstate(invalid="ignore", divide="ignore"):
            p_curve = f1_val * r_range / (2 * r_range - f1_val)
        valid = (p_curve >= 0.5) & (p_curve <= 1.02)
        ax.plot(r_range[valid], p_curve[valid], color="lightgray", linewidth=0.8, zorder=0)
        mid = np.searchsorted(r_range[valid], (r_range[valid].min() + r_range[valid].max()) / 2)
        if mid < len(r_range[valid]):
            ax.text(r_range[valid][mid], p_curve[valid][mid] + 0.005, f"F1={f1_val}",
                    fontsize=6, color="gray", ha="center")

    markers = {"speech": "o", "music": "s", "inactive": "^"}
    for model in data:
        for cls, marker in markers.items():
            pt = data[model]["per_class"].get(cls)
            if pt:
                ax.scatter(pt["r"], pt["p"], color=COLORS[model], marker=marker,
                           s=70, zorder=3, edgecolors="white", linewidths=0.5)
                ax.annotate(f"{MODEL_SHORT[model]}", (pt["r"], pt["p"]),
                            textcoords="offset points", xytext=(4, 2), fontsize=6, color=COLORS[model])

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=COLORS[m], marker="o", linestyle="None", markersize=6, label=MODEL_SHORT[m])
        for m in data
    ] + [
        Line2D([0], [0], color="gray", marker="o", linestyle="None", markersize=6, label="Speech (circle)"),
        Line2D([0], [0], color="gray", marker="s", linestyle="None", markersize=6, label="Music (square)"),
        Line2D([0], [0], color="gray", marker="^", linestyle="None", markersize=6, label="Inactive (triangle)"),
    ]
    ax.legend(handles=legend_els, fontsize=6, ncol=1)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall (3-class)")


def plot_inference_time(ax, data):
    labels = list(data.keys())
    times = [data[m]["time_per_frame"] or 0 for m in labels]
    bars = ax.bar(range(len(labels)), times, color=[COLORS[m] for m in labels], alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in labels])
    ax.set_ylabel("ms / frame")
    ax.set_title("Inference Time per Frame")
    for bar, t in zip(bars, times):
        if t > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{t:.2f}", ha="center", va="bottom", fontsize=8)


def plot_confusion_matrix(ax, cm, classes, title):
    if cm is None:
        ax.set_visible(False)
        return
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cm_norm[i, j] > 0.5 else "black")


def plot_subclass_group(ax, data, subclass_keys, short_names, title):
    models = list(data.keys())
    n_subs = len(subclass_keys)
    n_models = len(models)
    total_height = 0.72
    bar_h = total_height / n_models
    y = np.arange(n_subs)
    for i, model in enumerate(models):
        vals = [data[model]["subclasses"].get(s, {}).get("f1", 0) for s in subclass_keys]
        offset = (i - n_models / 2 + 0.5) * bar_h
        ax.barh(y + offset, vals, bar_h * 0.9, color=COLORS[model], alpha=0.85, label=MODEL_SHORT[model])
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1")
    ax.set_title(title)
    ax.legend(fontsize=7)


def main():
    data = {}
    for model in MODELS:
        path = RESULTS_DIR / f"{model}.eval"
        if path.exists():
            data[model] = parse_eval(path)

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle("Model Evaluation Results", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    # Row 1: Macro F1 | P/R scatter | Inference time
    plot_macro_f1(fig.add_subplot(gs[0, :2]), data)
    plot_pr_scatter(fig.add_subplot(gs[0, 2]), data)
    plot_inference_time(fig.add_subplot(gs[0, 3]), data)

    # Row 2: 3-class confusion matrices, one per model
    cm_classes = ["Speech", "Music", "Inactive"]
    for i, model in enumerate(data):
        plot_confusion_matrix(
            fig.add_subplot(gs[1, i]),
            data[model]["cm"],
            cm_classes,
            f"{MODEL_SHORT[model]} — 3-class CM",
        )

    # Row 3: Speech subclasses | Music subclasses
    plot_subclass_group(fig.add_subplot(gs[2, :2]), data, SPEECH_SUBS, SPEECH_SHORT, "Speech Subclasses — F1 by Model")
    plot_subclass_group(fig.add_subplot(gs[2, 2:]), data, MUSIC_SUBS, MUSIC_SHORT, "Music Subclasses — F1 by Model")

    out = RESULTS_DIR / "results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
