"""Visualize evaluation results for all models."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["tcn", "decision_tree", "svm", "gmm"]
COLORS = {"tcn": "#2196F3", "decision_tree": "#4CAF50", "svm": "#FF9800", "gmm": "#9C27B0"}
MODEL_SHORT = {"tcn": "TCN", "decision_tree": "DT", "svm": "SVM", "gmm": "GMM"}

SPEECH_SUBS = [
    "speech_clean", "speech_dirty", "speech_multispeaker",
    "speech_msom", "speech_noisy", "speech_som",
]
MUSIC_SUBS = [
    "music_acapella", "music_electronic", "music_folk", "music_hip-hop",
    "music_instrumental", "music_pop", "music_rock",
]
SPEECH_SHORT = ["clean", "dirty", "multi", "msom", "noisy", "som"]
MUSIC_SHORT = ["acapella", "electronic", "folk", "hip-hop", "instrumental", "pop", "rock"]


def parse_eval(path: Path) -> dict:
    text = path.read_text()

    def macro_f1(section: str) -> float | None:
        m = re.search(r"Macro\s+([\d.]+)", section)
        return float(m.group(1)) if m else None

    eval_section = re.search(r"──.*?valuation.*?── By subclass", text, re.DOTALL)
    subclass_section = re.search(r"── By subclass.*", text, re.DOTALL)

    time_m = re.search(r"(?:ms/frame|Time/frame)\s*:\s*([\d.]+)", text)
    time_per_frame = float(time_m.group(1)) if time_m else None

    device_m = re.search(r"Device\s*:\s*(.+)", text)
    device = device_m.group(1).strip() if device_m else None

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
            m = re.match(r"\s+(\S+)\s+([\d.]+)\s*$", line)
            if m:
                subclasses[m.group(1)] = {"r": float(m.group(2))}

    return {
        "macro_f1": macro_f1(eval_section.group()) if eval_section else None,
        "per_class": per_class,
        "cm": cm,
        "time_per_frame": time_per_frame,
        "device": device,
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
    ax.set_title("Macro F1")
    ax.set_ylabel("Macro F1")
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)


def plot_pr_scatter(ax, data):
    # Determine axis range from actual data
    all_r = [pt["r"] for m in data.values() for pt in m["per_class"].values()]
    all_p = [pt["p"] for m in data.values() for pt in m["per_class"].values()]
    lo = max(0.0, min(min(all_r), min(all_p)) - 0.05)
    lo = round(lo * 10) / 10  # snap to 0.1

    r_grid = np.linspace(lo, 1.0, 400)
    p_grid = np.linspace(lo, 1.0, 400)
    R, P = np.meshgrid(r_grid, p_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        F1 = np.where((P + R) > 0, 2 * P * R / (P + R), 0.0)

    # Filled F1 contour background
    levels = np.linspace(lo, 1.0, 200)
    ax.contourf(R, P, F1, levels=levels, cmap="YlOrRd_r", alpha=0.25, zorder=0)

    # Labeled iso-F1 lines
    iso_vals = [v for v in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99] if v >= lo]
    cs = ax.contour(R, P, F1, levels=iso_vals, colors="gray", linewidths=0.7, zorder=1, alpha=0.6)
    ax.clabel(cs, fmt=lambda v: f"F1={v:.2f}", fontsize=6, inline=True)

    markers = {"speech": "o", "music": "s", "inactive": "^"}
    for model in data:
        for cls, marker in markers.items():
            pt = data[model]["per_class"].get(cls)
            if pt:
                ax.scatter(pt["r"], pt["p"], color=COLORS[model], marker=marker,
                           s=70, zorder=3, edgecolors="white", linewidths=0.5)
                ax.annotate(MODEL_SHORT[model], (pt["r"], pt["p"]),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=6, color=COLORS[model])

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=COLORS[m], marker="o", linestyle="None", markersize=6, label=MODEL_SHORT[m])
        for m in data
    ] + [
        Line2D([0], [0], color="gray", marker="o", linestyle="None", markersize=6, label="Speech"),
        Line2D([0], [0], color="gray", marker="s", linestyle="None", markersize=6, label="Music"),
        Line2D([0], [0], color="gray", marker="^", linestyle="None", markersize=6, label="Inactive"),
    ]
    ax.legend(handles=legend_els, fontsize=6, ncol=1)
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")


def _is_gpu(device: str | None) -> bool:
    if not device:
        return False
    low = device.lower()
    return any(k in low for k in ("nvidia", "cuda", "geforce", "quadro", "tesla", "amd instinct", "radeon"))


def plot_inference_time(ax, data):
    labels = list(data.keys())
    times = [data[m]["time_per_frame"] or 0 for m in labels]
    devices = [data[m].get("device") for m in labels]
    gpu_flags = [_is_gpu(d) for d in devices]

    gpu_items = [(m, times[i]) for i, m in enumerate(labels) if gpu_flags[i] and times[i] > 0]
    cpu_items = [(m, times[i]) for i, m in enumerate(labels) if not gpu_flags[i] and times[i] > 0]

    fig = ax.figure
    spec = ax.get_subplotspec()
    # Re-purpose `ax` as an invisible container for the title and device caption.
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("none")
    ax.set_title("Process Time per Frame")

    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.0)
    ax_gpu = fig.add_subplot(inner[0, 0])
    ax_cpu = fig.add_subplot(inner[0, 1])

    def _draw(sub_ax, items, group_label):
        if not items:
            sub_ax.set_visible(False)
            return
        idx = list(range(len(items)))
        ts = [t for _, t in items]
        bars = sub_ax.bar(
            idx, ts,
            color=[COLORS[m] for m, _ in items],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        sub_ax.set_xticks(idx)
        sub_ax.set_xticklabels([MODEL_SHORT[m] for m, _ in items])
        sub_ax.set_ylim(0, max(ts) * 1.25)
        sub_ax.text(
            0.5, 0.97, group_label,
            transform=sub_ax.transAxes, ha="center", va="top",
            fontsize=8, color="gray",
        )
        for bar, t in zip(bars, ts):
            sub_ax.text(bar.get_x() + bar.get_width() / 2, t,
                        f"{t:.3f}", ha="center", va="bottom", fontsize=8)

    _draw(ax_gpu, gpu_items, "GPU")
    _draw(ax_cpu, cpu_items, "CPU")

    ax_gpu.set_ylabel("ms / frame")
    ax_cpu.set_ylabel("ms / frame")
    ax_cpu.yaxis.set_label_position("right")
    ax_cpu.yaxis.tick_right()
    ax_gpu.spines["right"].set_visible(False)
    ax_cpu.spines["left"].set_visible(False)
    # Dashed separator at the shared edge between the two subplots.
    from matplotlib.lines import Line2D
    ax_gpu.add_line(Line2D(
        [1, 1], [0, 1], transform=ax_gpu.transAxes,
        color="gray", linestyle="--", linewidth=1, clip_on=False, zorder=10,
    ))

    caption = "\n".join(
        f"{MODEL_SHORT[m]}: {devices[i] or 'unknown'}"
        for i, m in enumerate(labels)
    )
    ax.text(
        0.5, -0.18, caption,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=6, color="gray",
    )


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
        vals = [data[model]["subclasses"].get(s, {}).get("r", 0) for s in subclass_keys]
        offset = (i - n_models / 2 + 0.5) * bar_h
        ax.barh(y + offset, vals, bar_h * 0.9, color=COLORS[model], alpha=0.85, label=MODEL_SHORT[model])
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_title(title)
    ax.legend(fontsize=7)


def _save_solo(plot_fn, *args, path, figsize=(8, 6), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, *args, **kwargs)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def main():
    data = {}
    for model in MODELS:
        path = RESULTS_DIR / f"{model}.eval"
        if path.exists():
            data[model] = parse_eval(path)

    cm_classes = ["Speech", "Music", "Inactive"]

    # Solo saves
    graphs_dir = RESULTS_DIR / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    _save_solo(plot_macro_f1, data, path=graphs_dir / "macro_f1.png", figsize=(7, 5))
    _save_solo(plot_pr_scatter, data, path=graphs_dir / "pr_scatter.png")
    _save_solo(plot_inference_time, data, path=graphs_dir / "inference_time.png", figsize=(6, 5))
    for model in data:
        _save_solo(
            plot_confusion_matrix, data[model]["cm"], cm_classes,
            f"{MODEL_SHORT[model]} — Confusion Matrix",
            path=graphs_dir / f"cm_{model}.png", figsize=(5, 4),
        )
    _save_solo(plot_subclass_group, data, SPEECH_SUBS, SPEECH_SHORT,
               "Speech Subclasses — Recall by Model", path=graphs_dir / "speech_subclasses.png", figsize=(9, 6))
    _save_solo(plot_subclass_group, data, MUSIC_SUBS, MUSIC_SHORT,
               "Music Subclasses — Recall by Model", path=graphs_dir / "music_subclasses.png", figsize=(9, 6))

    # Combined
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle("Model Evaluation Results", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    plot_macro_f1(fig.add_subplot(gs[0, :2]), data)
    plot_pr_scatter(fig.add_subplot(gs[0, 2]), data)
    plot_inference_time(fig.add_subplot(gs[0, 3]), data)

    for i, model in enumerate(data):
        plot_confusion_matrix(
            fig.add_subplot(gs[1, i]),
            data[model]["cm"],
            cm_classes,
            f"{MODEL_SHORT[model]} — Confusion Matrix",
        )

    plot_subclass_group(fig.add_subplot(gs[2, :2]), data, SPEECH_SUBS, SPEECH_SHORT, "Speech Subclasses — Recall by Model")
    plot_subclass_group(fig.add_subplot(gs[2, 2:]), data, MUSIC_SUBS, MUSIC_SHORT, "Music Subclasses — Recall by Model")

    out = RESULTS_DIR / "results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
