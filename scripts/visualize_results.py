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

    def macro_f1_and_ci(section: str):
        m = re.search(r"Macro F1\s*:\s*([\d.]+)(?:\s*\[([\d.]+)\s*,\s*([\d.]+)\])?", section)
        if not m:
            return None, None
        pt = float(m.group(1))
        ci = (float(m.group(2)), float(m.group(3))) if m.group(2) else None
        return pt, ci

    def single_float(section: str, key: str) -> float | None:
        m = re.search(rf"{key}\s*:\s*([\d.]+)", section)
        return float(m.group(1)) if m else None

    eval_section = re.search(r"──.*?valuation.*?── By subclass", text, re.DOTALL)
    subclass_section = re.search(r"── By subclass.*", text, re.DOTALL)

    time_m = re.search(r"(?:ms/frame|Time/frame)\s*:\s*([\d.]+)", text)
    time_per_frame = float(time_m.group(1)) if time_m else None

    device_m = re.search(r"Device\s*:\s*(.+)", text)
    device = device_m.group(1).strip() if device_m else None

    per_class = {}
    per_class_ci = {}
    if eval_section:
        for cls in ["Speech", "Music", "Inactive"]:
            m = re.search(
                rf"{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+\[([\d.]+),\s*([\d.]+)\])?",
                eval_section.group(),
            )
            if m:
                per_class[cls.lower()] = {
                    "f1": float(m.group(1)), "p": float(m.group(2)), "r": float(m.group(3)),
                }
                if m.group(4):
                    per_class_ci[cls.lower()] = (float(m.group(4)), float(m.group(5)))

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
            m = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if m:
                subclasses[m.group(1)] = {"f1": float(m.group(2))}

    macro_pt, macro_ci = macro_f1_and_ci(eval_section.group()) if eval_section else (None, None)
    weighted = single_float(eval_section.group(), "Weighted F1") if eval_section else None
    acc = single_float(eval_section.group(), "Accuracy") if eval_section else None
    auroc = single_float(eval_section.group(), "Macro AUROC") if eval_section else None

    return {
        "macro_f1": macro_pt,
        "macro_f1_ci": macro_ci,
        "weighted_f1": weighted,
        "accuracy": acc,
        "macro_auroc": auroc,
        "per_class": per_class,
        "per_class_ci": per_class_ci,
        "cm": cm,
        "time_per_frame": time_per_frame,
        "device": device,
        "subclasses": subclasses,
    }


def plot_macro_f1(ax, data):
    labels = list(data.keys())
    x = np.arange(len(labels))
    f1_vals = [data[m]["macro_f1"] or 0 for m in labels]
    # Asymmetric error bars from bootstrap CI when available.
    err_lo, err_hi = [], []
    have_ci = False
    for m, pt in zip(labels, f1_vals):
        ci = data[m].get("macro_f1_ci")
        if ci:
            err_lo.append(max(0.0, pt - ci[0]))
            err_hi.append(max(0.0, ci[1] - pt))
            have_ci = True
        else:
            err_lo.append(0.0)
            err_hi.append(0.0)
    bars = ax.bar(x, f1_vals, 0.5, color=[COLORS[m] for m in labels], alpha=0.9)
    if have_ci:
        ax.errorbar(
            x, f1_vals, yerr=[err_lo, err_hi],
            fmt="none", ecolor="black", capsize=4, lw=1.0, alpha=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in labels])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Macro F1" + (" (95% CI)" if have_ci else ""))
    ax.set_ylabel("Macro F1")
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            inside = h > 0.975
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h - 0.01 if inside else h + 0.01,
                    f"{h:.2f}", ha="center",
                    va="top" if inside else "bottom",
                    fontsize=7,
                    color="white" if inside else "black")


def plot_metric_table(ax, data):
    """Compact text table: Macro F1 (± CI), Weighted F1, Accuracy, Macro AUROC."""
    ax.axis("off")
    models = list(data.keys())
    headers = ["Model", "Macro F1", "Weighted F1", "Accuracy", "Macro AUROC"]
    rows = []
    for m in models:
        d = data[m]
        macro = d.get("macro_f1")
        ci = d.get("macro_f1_ci")
        if macro is None:
            macro_s = "—"
        elif ci:
            macro_s = f"{macro:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        else:
            macro_s = f"{macro:.4f}"
        rows.append([
            MODEL_SHORT[m],
            macro_s,
            f"{d['weighted_f1']:.4f}" if d.get("weighted_f1") is not None else "—",
            f"{d['accuracy']:.4f}" if d.get("accuracy") is not None else "—",
            f"{d['macro_auroc']:.4f}" if d.get("macro_auroc") is not None else "—",
        ])
    tbl = ax.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    for i, m in enumerate(models):
        tbl[(i + 1, 0)].set_facecolor(COLORS[m])
        tbl[(i + 1, 0)].set_alpha(0.4)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#e0e0e0")
        tbl[(0, j)].set_text_props(weight="bold")
    ax.set_title("Headline Metrics")


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
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
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
        vals = [data[model]["subclasses"].get(s, {}).get("f1", 0) for s in subclass_keys]
        offset = (i - n_models / 2 + 0.5) * bar_h
        ax.barh(y + offset, vals, bar_h * 0.9, color=COLORS[model], alpha=0.85, label=MODEL_SHORT[model])
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1")
    ax.set_title(title)
    ax.legend(fontsize=7)


EVAL_LABELS = [-1, 1, 2]
EVAL_LABEL_NAMES = {-1: "Speech", 1: "Music", 2: "Inactive"}


def load_scores(results_dir: Path) -> dict:
    """Load *_scores.npz files. Returns {model_name: {"y_true": ..., "y_scores": ...}}."""
    data = {}
    for npz_path in sorted(results_dir.glob("*_scores.npz")):
        model = npz_path.stem.replace("_scores", "")
        d = np.load(npz_path)
        data[model] = {"y_true": d["y_true"], "y_scores": d["y_scores"]}
    return data


def plot_roc_curves(ax, scores_data: dict, class_idx: int, class_label: int):
    """One-vs-rest ROC curve for a single class. All models on one plot."""
    from sklearn.metrics import roc_curve, roc_auc_score

    for model, sd in scores_data.items():
        y_binary = (sd["y_true"] == class_label).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            continue
        fpr, tpr, _ = roc_curve(y_binary, sd["y_scores"][:, class_idx])
        auc = roc_auc_score(y_binary, sd["y_scores"][:, class_idx])
        ax.plot(fpr, tpr, color=COLORS.get(model, "gray"), lw=1.5,
                label=f"{MODEL_SHORT.get(model, model)} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {EVAL_LABEL_NAMES[class_label]} vs Rest")
    ax.legend(fontsize=7)


def plot_det_curves(ax, scores_data: dict, class_idx: int, class_label: int):
    """One-vs-rest DET curve for a single class. All models on one plot."""
    from sklearn.metrics import det_curve

    for model, sd in scores_data.items():
        y_binary = (sd["y_true"] == class_label).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            continue
        fpr, fnr, _ = det_curve(y_binary, sd["y_scores"][:, class_idx])
        ax.plot(fpr, fnr, color=COLORS.get(model, "gray"), lw=1.5,
                label=MODEL_SHORT.get(model, model))
    ax.set_xlabel("FPR")
    ax.set_ylabel("FNR (Miss Rate)")
    ax.set_title(f"DET — {EVAL_LABEL_NAMES[class_label]} vs Rest")
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
    _save_solo(plot_metric_table, data, path=graphs_dir / "metric_table.png", figsize=(10, 3))
    _save_solo(plot_pr_scatter, data, path=graphs_dir / "pr_scatter.png")
    _save_solo(plot_inference_time, data, path=graphs_dir / "inference_time.png", figsize=(6, 5))
    for model in data:
        _save_solo(
            plot_confusion_matrix, data[model]["cm"], cm_classes,
            f"{MODEL_SHORT[model]} — Confusion Matrix",
            path=graphs_dir / f"cm_{model}.png", figsize=(5, 4),
        )
    _save_solo(plot_subclass_group, data, SPEECH_SUBS, SPEECH_SHORT,
               "Speech Subclasses — F1 by Model", path=graphs_dir / "speech_subclasses.png", figsize=(9, 6))
    _save_solo(plot_subclass_group, data, MUSIC_SUBS, MUSIC_SHORT,
               "Music Subclasses — F1 by Model", path=graphs_dir / "music_subclasses.png", figsize=(9, 6))

    # ROC / DET curves
    scores_data = load_scores(RESULTS_DIR)
    if scores_data:
        fig_rd, axes_rd = plt.subplots(2, 3, figsize=(15, 9))
        fig_rd.suptitle("ROC & DET Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
        for j, (idx, lbl) in enumerate(zip(range(3), EVAL_LABELS)):
            plot_roc_curves(axes_rd[0, j], scores_data, idx, lbl)
            plot_det_curves(axes_rd[1, j], scores_data, idx, lbl)
        fig_rd.tight_layout(rect=[0, 0, 1, 0.95])
        rd_path = graphs_dir / "roc_det.png"
        fig_rd.savefig(rd_path, dpi=150, bbox_inches="tight")
        plt.close(fig_rd)
        print(f"Saved → {rd_path}")

    # Combined
    fig = plt.figure(figsize=(18, 19))
    fig.suptitle("Model Evaluation Results", fontsize=14, fontweight="bold")
    gs = GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38, height_ratios=[0.6, 1.0, 1.0, 1.0])

    plot_metric_table(fig.add_subplot(gs[0, :]), data)

    plot_macro_f1(fig.add_subplot(gs[1, :2]), data)
    plot_pr_scatter(fig.add_subplot(gs[1, 2]), data)
    plot_inference_time(fig.add_subplot(gs[1, 3]), data)

    for i, model in enumerate(data):
        plot_confusion_matrix(
            fig.add_subplot(gs[2, i]),
            data[model]["cm"],
            cm_classes,
            f"{MODEL_SHORT[model]} — Confusion Matrix",
        )

    plot_subclass_group(fig.add_subplot(gs[3, :2]), data, SPEECH_SUBS, SPEECH_SHORT, "Speech Subclasses — F1 by Model")
    plot_subclass_group(fig.add_subplot(gs[3, 2:]), data, MUSIC_SUBS, MUSIC_SHORT, "Music Subclasses — F1 by Model")

    out = RESULTS_DIR / "results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
