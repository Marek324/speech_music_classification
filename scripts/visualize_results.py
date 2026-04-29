"""Visualize evaluation results for all models."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["tcn_lstm", "small_tcn", "smaller_tcn", "tcn", "decision_tree", "svm", "gmm"]
NN_MODELS = ["tcn_lstm", "small_tcn", "smaller_tcn", "tcn"]
COLORS = {"tcn_lstm": "#E91E63", "small_tcn": "#009688", "smaller_tcn": "#4DB6AC", "tcn": "#2196F3", "decision_tree": "#4CAF50", "svm": "#FF9800", "gmm": "#9C27B0"}
MODEL_SHORT = {"tcn_lstm": "TCN+LSTM", "small_tcn": "SmallTCN", "smaller_tcn": "SmallerTCN", "tcn": "TCN", "decision_tree": "DT", "svm": "SVM", "gmm": "GMM"}

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
ALL_SUBS = SPEECH_SUBS + MUSIC_SUBS + ["noise"]
ALL_SHORT = SPEECH_SHORT + MUSIC_SHORT + ["noise"]


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
    """Load *_scores.npz files for known models. Returns {model_name: {"y_true": ..., "y_scores": ...}}."""
    data = {}
    for model in MODELS:
        npz_path = results_dir / f"{model}_scores.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        data[model] = {"y_true": d["y_true"], "y_scores": d["y_scores"]}
    return data


def _has_continuous_scores(sd: dict, class_idx: int, min_unique: int = 50) -> bool:
    """ROC/DET only meaningful when the score column has a usable threshold range."""
    return len(np.unique(sd["y_scores"][:, class_idx])) >= min_unique


def plot_roc_curves(ax, scores_data: dict, class_idx: int, class_label: int):
    """One-vs-rest ROC curve for a single class. All models on one plot."""
    from sklearn.metrics import roc_curve, roc_auc_score

    for model, sd in scores_data.items():
        y_binary = (sd["y_true"] == class_label).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            continue
        if not _has_continuous_scores(sd, class_idx):
            continue  # e.g. decision tree has 3-5 unique probs → degenerate curve
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
    ax.legend(fontsize=7, loc="lower right")


def plot_det_curves(ax, scores_data: dict, class_idx: int, class_label: int):
    """One-vs-rest DET curve for a single class. Normal-deviate (probit) axes."""
    from sklearn.metrics import det_curve
    from scipy.stats import norm

    # Clip to the visible probit range so off-screen tails don't connect through
    # the plot interior. xlim is ±3σ → fpr/fnr ∈ [norm.cdf(-3.2), norm.cdf(3.2)].
    lo = norm.cdf(-3.2)
    hi = norm.cdf(3.2)
    for model, sd in scores_data.items():
        y_binary = (sd["y_true"] == class_label).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            continue
        if not _has_continuous_scores(sd, class_idx):
            continue
        fpr, fnr, _ = det_curve(y_binary, sd["y_scores"][:, class_idx])
        fpr = np.clip(fpr, lo, hi)
        fnr = np.clip(fnr, lo, hi)
        ax.plot(norm.ppf(fpr), norm.ppf(fnr), color=COLORS.get(model, "gray"),
                lw=1.5, label=MODEL_SHORT.get(model, model))
    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_vals = norm.ppf(ticks)
    tick_labels = [
        f"{t:.0%}" if (100 * t).is_integer() else f"{t:.1%}" for t in ticks
    ]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("FNR (Miss Rate)")
    ax.set_title(f"DET — {EVAL_LABEL_NAMES[class_label]} vs Rest")
    ax.legend(fontsize=7, loc="upper right")


def plot_per_class_f1(ax, data):
    """Grouped bars: per-class F1, one bar group per class, one colored bar per model."""
    models = list(data.keys())
    classes = ["speech", "music", "inactive"]
    class_labels = ["Speech", "Music", "Inactive"]
    n_models = len(models)
    x = np.arange(len(classes))
    width = 0.8 / n_models
    for i, model in enumerate(models):
        vals = [data[model]["per_class"].get(c, {}).get("f1", 0) for c in classes]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.95,
                      color=COLORS[model], alpha=0.9, label=MODEL_SHORT[model])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Per-Class F1 by Model")
    ax.legend(fontsize=8, ncol=n_models, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def plot_subclass_heatmap(ax, data):
    """Heatmap: rows=models, cols=all 14 subclasses (speech | music | inactive). Cells = F1."""
    models = list(data.keys())
    values = np.full((len(models), len(ALL_SUBS)), np.nan)
    for i, model in enumerate(models):
        for j, sub in enumerate(ALL_SUBS):
            f1 = data[model]["subclasses"].get(sub, {}).get("f1")
            if f1 is not None:
                values[i, j] = f1

    # Clamp low end of colormap so differences near ceiling are still visible.
    vmin = np.nanmin(values)
    vmin_floor = max(0.0, min(vmin - 0.02, 0.7))
    im = ax.imshow(values, cmap="RdYlGn", vmin=vmin_floor, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(ALL_SUBS)))
    ax.set_xticklabels(ALL_SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_SHORT[m] for m in models], fontsize=9)

    speech_end = len(SPEECH_SUBS) - 0.5
    music_end = len(SPEECH_SUBS) + len(MUSIC_SUBS) - 0.5
    for xv in (speech_end, music_end):
        ax.axvline(xv, color="black", linewidth=1.2)

    ax.text(len(SPEECH_SUBS) / 2 - 0.5, -0.9, "Speech",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(len(SPEECH_SUBS) + len(MUSIC_SUBS) / 2 - 0.5, -0.9, "Music",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(len(SPEECH_SUBS) + len(MUSIC_SUBS) - 0.5, -0.9, "Inactive",
            ha="center", fontsize=10, fontweight="bold")

    mid = (vmin_floor + 1.0) / 2
    for i in range(len(models)):
        for j in range(len(ALL_SUBS)):
            v = values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v < mid - 0.05 else "black")

    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("F1", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Subclass F1 — all models × all subclasses")


def plot_subclass_dumbbell(ax, data, model_a: str, model_b: str):
    """Dumbbell per-subclass F1: `model_a` vs `model_b`, sorted by Δ = f1(a) − f1(b).
    Line color indicates winner (green if model_a wins, red if model_b wins)."""
    if model_a not in data or model_b not in data:
        ax.set_visible(False)
        return

    group_for = (
        {s: "sp" for s in SPEECH_SUBS}
        | {s: "mu" for s in MUSIC_SUBS}
        | {"noise": "in"}
    )
    short_for = dict(zip(ALL_SUBS, ALL_SHORT))

    rows = []
    for sub in ALL_SUBS:
        a_f1 = data[model_a]["subclasses"].get(sub, {}).get("f1")
        b_f1 = data[model_b]["subclasses"].get(sub, {}).get("f1")
        if a_f1 is None or b_f1 is None:
            continue
        rows.append((sub, b_f1, a_f1, a_f1 - b_f1))

    rows.sort(key=lambda r: r[3], reverse=True)

    n = len(rows)
    y = np.arange(n)
    win_color = "#2E7D32"
    loss_color = "#C62828"
    a_color = COLORS[model_a]
    b_color = COLORS[model_b]
    a_name = MODEL_SHORT[model_a]
    b_name = MODEL_SHORT[model_b]

    for i, (_, b_val, a_val, d) in enumerate(rows):
        line_color = win_color if d >= 0 else loss_color
        ax.plot([b_val, a_val], [i, i], color=line_color, lw=2.2, alpha=0.75, zorder=1)
        ax.scatter([b_val], [i], color=b_color, s=70, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.scatter([a_val], [i], color=a_color, s=70, zorder=3,
                   edgecolor="white", linewidth=0.8)
        sign = "+" if d >= 0 else ""
        right = max(b_val, a_val)
        ax.text(right + 0.003, i, f"{sign}{d*100:.1f} pp",
                va="center", ha="left", fontsize=7,
                color=line_color, fontweight="bold")

    ax.set_yticks(y)
    labels = [f"{group_for[s]} / {short_for[s]}" for s, _, _, _ in rows]
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()

    all_vals = [v for r in rows for v in (r[1], r[2])]
    lo = max(0.5, min(all_vals) - 0.01)
    ax.set_xlim(lo, 1.01)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("F1")
    ax.set_title(f"Subclass F1 — {a_name} vs {b_name}  (sorted by Δ)")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=b_color, marker="o", linestyle="None", markersize=7, label=b_name),
        Line2D([0], [0], color=a_color, marker="o", linestyle="None", markersize=7, label=a_name),
        Line2D([0], [0], color=win_color, lw=2.5, label=f"{a_name} better"),
        Line2D([0], [0], color=loss_color, lw=2.5, label=f"{a_name} worse"),
    ]
    ax.legend(handles=legend_els, fontsize=7, loc="lower left", framealpha=0.9)


def plot_cm_delta(ax, other_cm, ref_cm, classes, title, vmax_shared=None, show_ylabel=True):
    """Row-normalized CM delta vs reference model, sign-flipped so positive always = worse than reference."""
    if other_cm is None or ref_cm is None:
        ax.set_visible(False)
        return

    def norm(cm):
        rs = cm.sum(axis=1, keepdims=True)
        return np.where(rs > 0, cm / rs, 0.0)

    o = norm(ref_cm)
    x = norm(other_cm)
    I_mask = np.eye(len(classes), dtype=bool)
    err_delta = np.where(I_mask, o - x, x - o)  # positive ⇒ this model does worse at that cell

    vmax = vmax_shared if vmax_shared is not None else max(np.abs(err_delta).max(), 1e-6)
    im = ax.imshow(err_delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    if show_ylabel:
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_ylabel("True", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = err_delta[i, j]
            sign = "+" if v > 0 else ""
            ax.text(j, i, f"{sign}{v*100:.1f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(v) > vmax * 0.65 else "black")
    return im


def plot_reliability(ax, scores_data: dict, n_bins: int = 15):
    """Reliability diagram: predicted max-prob confidence vs empirical accuracy, per model."""
    bins = np.linspace(0, 1, n_bins + 1)
    label_arr = np.array(EVAL_LABELS)

    for model, sd in scores_data.items():
        y_true = sd["y_true"]
        y_scores = sd["y_scores"].astype(float)
        row_sums = y_scores.sum(axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=1e-2):
            y_scores = y_scores / np.clip(row_sums, 1e-10, None)

        pred_idx = np.argmax(y_scores, axis=1)
        pred_label = label_arr[pred_idx]
        conf = y_scores[np.arange(len(y_scores)), pred_idx]
        correct = (pred_label == y_true).astype(float)

        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
            if mask.sum() >= 50:
                xs.append(conf[mask].mean())
                ys.append(correct[mask].mean())

        # Expected Calibration Error (ECE): weighted |acc - conf| across bins
        ece = 0.0
        total = len(conf)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
            if mask.sum() > 0:
                ece += (mask.sum() / total) * abs(conf[mask].mean() - correct[mask].mean())

        ax.plot(xs, ys, "o-", color=COLORS.get(model, "gray"),
                label=f"{MODEL_SHORT.get(model, model)} (ECE={ece:.3f})",
                markersize=5, lw=1.6)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=0.8, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence (max prob)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")


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
    _save_solo(plot_macro_f1, data, path=graphs_dir / "macro_f1.svg", figsize=(7, 5))
    _save_solo(plot_metric_table, data, path=graphs_dir / "metric_table.svg", figsize=(10, 3))
    _save_solo(plot_pr_scatter, data, path=graphs_dir / "pr_scatter.svg")
    _save_solo(plot_inference_time, data, path=graphs_dir / "inference_time.svg", figsize=(6, 5))
    for model in data:
        _save_solo(
            plot_confusion_matrix, data[model]["cm"], cm_classes,
            f"{MODEL_SHORT[model]} — Confusion Matrix",
            path=graphs_dir / f"cm_{model}.svg", figsize=(5, 4),
        )
    _save_solo(plot_subclass_group, data, SPEECH_SUBS, SPEECH_SHORT,
               "Speech Subclasses — F1 by Model", path=graphs_dir / "speech_subclasses.svg", figsize=(9, 6))
    _save_solo(plot_subclass_group, data, MUSIC_SUBS, MUSIC_SHORT,
               "Music Subclasses — F1 by Model", path=graphs_dir / "music_subclasses.svg", figsize=(9, 6))

    _save_solo(plot_per_class_f1, data, path=graphs_dir / "per_class_f1.svg", figsize=(8, 5))
    _save_solo(plot_subclass_heatmap, data, path=graphs_dir / "subclass_heatmap.svg", figsize=(13, 4))

    # Confusion matrix deltas vs TCN+LSTM
    if "tcn_lstm" in data and data["tcn_lstm"]["cm"] is not None:
        others = [m for m in data if m != "tcn_lstm" and data[m]["cm"] is not None]
        if others:
            # Shared vmax so colors are comparable across the 4 panels.
            def _norm(cm):
                rs = cm.sum(axis=1, keepdims=True)
                return np.where(rs > 0, cm / rs, 0.0)
            o = _norm(data["tcn_lstm"]["cm"])
            I_mask = np.eye(len(cm_classes), dtype=bool)
            vmax_shared = 0.0
            for m in others:
                x = _norm(data[m]["cm"])
                d = np.where(I_mask, o - x, x - o)
                vmax_shared = max(vmax_shared, np.abs(d).max())
            vmax_shared = max(vmax_shared, 1e-3)

            fig_d, axes_d = plt.subplots(1, len(others), figsize=(4.2 * len(others), 4.2))
            if len(others) == 1:
                axes_d = [axes_d]
            fig_d.suptitle("Confusion Matrix Δ vs TCN+LSTM  (red = worse than TCN+LSTM, values in pp)",
                           fontsize=12, fontweight="bold")
            im = None
            for k, (ax_d, m) in enumerate(zip(axes_d, others)):
                im = plot_cm_delta(ax_d, data[m]["cm"], data["tcn_lstm"]["cm"], cm_classes,
                                   f"{MODEL_SHORT[m]} − TCN+LSTM", vmax_shared=vmax_shared,
                                   show_ylabel=(k == 0))
            if im is not None:
                cbar = fig_d.colorbar(im, ax=axes_d, fraction=0.025, pad=0.02)
                cbar.set_label("Δ error (pp)", fontsize=9)
                ticks = cbar.get_ticks()
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{t*100:+.0f}" for t in ticks])
            d_path = graphs_dir / "cm_deltas.svg"
            fig_d.savefig(d_path, dpi=150, bbox_inches="tight")
            plt.close(fig_d)
            print(f"Saved → {d_path}")

    # Standalone TCN+LSTM vs TCN delta (TCN as reference; blue ⇒ TCN+LSTM better, red ⇒ TCN+LSTM worse)
    if "tcn_lstm" in data and "tcn" in data and data["tcn_lstm"]["cm"] is not None and data["tcn"]["cm"] is not None:
        fig_ot, ax_ot = plt.subplots(figsize=(6, 5))
        im_ot = plot_cm_delta(
            ax_ot, data["tcn_lstm"]["cm"], data["tcn"]["cm"], cm_classes,
            "TCN+LSTM vs TCN  (red = TCN+LSTM worse, blue = TCN+LSTM better)",
        )
        if im_ot is not None:
            cbar = fig_ot.colorbar(im_ot, ax=ax_ot, fraction=0.045, pad=0.04)
            cbar.set_label("Δ error (pp)", fontsize=9)
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t*100:+.1f}" for t in ticks])
        ot_path = graphs_dir / "cm_delta_tcn_lstm_vs_tcn.svg"
        fig_ot.savefig(ot_path, dpi=150, bbox_inches="tight")
        plt.close(fig_ot)
        print(f"Saved → {ot_path}")

    if "tcn_lstm" in data and "tcn" in data:
        _save_solo(plot_subclass_dumbbell, data, "tcn_lstm", "tcn",
                   path=graphs_dir / "subclass_dumbbell_tcn_lstm_vs_tcn.svg", figsize=(9, 6))

    # ROC / DET curves
    scores_data = load_scores(RESULTS_DIR)
    if scores_data:
        fig_rd, axes_rd = plt.subplots(2, 3, figsize=(15, 9))
        fig_rd.suptitle("ROC & DET Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
        for j, (idx, lbl) in enumerate(zip(range(3), EVAL_LABELS)):
            plot_roc_curves(axes_rd[0, j], scores_data, idx, lbl)
            plot_det_curves(axes_rd[1, j], scores_data, idx, lbl)
        fig_rd.tight_layout(rect=[0, 0, 1, 0.95])
        rd_path = graphs_dir / "roc_det.svg"
        fig_rd.savefig(rd_path, dpi=150, bbox_inches="tight")
        plt.close(fig_rd)
        print(f"Saved → {rd_path}")

        _save_solo(plot_reliability, scores_data,
                   path=graphs_dir / "reliability.svg", figsize=(7, 6))

    # ── NN-only comparison (TCN / SmallTCN / TCN+LSTM) ───────────────────────
    nn_data = {m: data[m] for m in NN_MODELS if m in data}
    if len(nn_data) >= 2:
        _save_solo(plot_macro_f1, nn_data,
                   path=graphs_dir / "nn_macro_f1.svg", figsize=(5, 5))
        _save_solo(plot_metric_table, nn_data,
                   path=graphs_dir / "nn_metric_table.svg", figsize=(10, 2.2))
        _save_solo(plot_per_class_f1, nn_data,
                   path=graphs_dir / "nn_per_class_f1.svg", figsize=(7, 5))
        _save_solo(plot_subclass_heatmap, nn_data,
                   path=graphs_dir / "nn_subclass_heatmap.svg", figsize=(13, 3))
        _save_solo(plot_subclass_group, nn_data, SPEECH_SUBS, SPEECH_SHORT,
                   "Speech Subclasses — NN Models",
                   path=graphs_dir / "nn_speech_subclasses.svg", figsize=(8, 5))
        _save_solo(plot_subclass_group, nn_data, MUSIC_SUBS, MUSIC_SHORT,
                   "Music Subclasses — NN Models",
                   path=graphs_dir / "nn_music_subclasses.svg", figsize=(8, 5))
        _save_solo(plot_inference_time, nn_data,
                   path=graphs_dir / "nn_inference_time.svg", figsize=(5, 5))

        # Three CMs side by side — shared normalization range (always 0..1)
        nn_cm_present = [m for m in nn_data if nn_data[m]["cm"] is not None]
        if nn_cm_present:
            fig_p, axes_p = plt.subplots(1, len(nn_cm_present),
                                         figsize=(4 * len(nn_cm_present), 4))
            if len(nn_cm_present) == 1:
                axes_p = [axes_p]
            for ax_p, m in zip(axes_p, nn_cm_present):
                plot_confusion_matrix(ax_p, nn_data[m]["cm"], cm_classes,
                                      f"{MODEL_SHORT[m]} — Confusion Matrix")
            fig_p.suptitle("NN Models — Confusion Matrices",
                           fontsize=12, fontweight="bold")
            fig_p.tight_layout()
            panel_path = graphs_dir / "nn_cm_panel.svg"
            fig_p.savefig(panel_path, dpi=150, bbox_inches="tight")
            plt.close(fig_p)
            print(f"Saved → {panel_path}")

        # CM deltas with TCN as reference: every NN variant vs TCN, shared vmax.
        if "tcn" in nn_data and nn_data["tcn"]["cm"] is not None:
            others = [m for m in NN_MODELS if m != "tcn"
                      and m in nn_data and nn_data[m]["cm"] is not None]
            if others:
                def _norm(cm):
                    rs = cm.sum(axis=1, keepdims=True)
                    return np.where(rs > 0, cm / rs, 0.0)
                ref_norm = _norm(nn_data["tcn"]["cm"])
                I_mask = np.eye(len(cm_classes), dtype=bool)
                vmax_shared = 0.0
                for m in others:
                    x = _norm(nn_data[m]["cm"])
                    d = np.where(I_mask, ref_norm - x, x - ref_norm)
                    vmax_shared = max(vmax_shared, np.abs(d).max())
                vmax_shared = max(vmax_shared, 1e-3)

                fig_nd, axes_nd = plt.subplots(1, len(others),
                                               figsize=(4.8 * len(others), 4.5))
                if len(others) == 1:
                    axes_nd = [axes_nd]
                fig_nd.suptitle(
                    "Confusion Matrix Δ vs TCN  (red = worse than TCN, values in pp)",
                    fontsize=12, fontweight="bold",
                )
                im = None
                for k, (ax_nd, m) in enumerate(zip(axes_nd, others)):
                    im = plot_cm_delta(
                        ax_nd, nn_data[m]["cm"], nn_data["tcn"]["cm"], cm_classes,
                        f"{MODEL_SHORT[m]} − TCN",
                        vmax_shared=vmax_shared, show_ylabel=(k == 0),
                    )
                if im is not None:
                    cbar = fig_nd.colorbar(im, ax=axes_nd, fraction=0.035, pad=0.02)
                    cbar.set_label("Δ error (pp)", fontsize=9)
                    ticks = cbar.get_ticks()
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels([f"{t*100:+.0f}" for t in ticks])
                nd_path = graphs_dir / "nn_cm_deltas_vs_tcn.svg"
                fig_nd.savefig(nd_path, dpi=150, bbox_inches="tight")
                plt.close(fig_nd)
                print(f"Saved → {nd_path}")

        # Complementary dumbbell mirror (TCN+LSTM vs TCN already emitted above).
        for m in NN_MODELS:
            if m in ("tcn", "tcn_lstm"):
                continue
            if m in nn_data and "tcn" in nn_data:
                _save_solo(plot_subclass_dumbbell, data, m, "tcn",
                           path=graphs_dir / f"subclass_dumbbell_{m}_vs_tcn.svg",
                           figsize=(9, 6))

    nn_scores_data = {m: scores_data[m] for m in NN_MODELS if m in scores_data}
    if len(nn_scores_data) >= 2:
        _save_solo(plot_reliability, nn_scores_data,
                   path=graphs_dir / "nn_reliability.svg", figsize=(7, 6))

        fig_nrd, axes_nrd = plt.subplots(2, 3, figsize=(15, 9))
        fig_nrd.suptitle("ROC & DET — NN Models (One-vs-Rest)",
                         fontsize=13, fontweight="bold")
        for j, (idx, lbl) in enumerate(zip(range(3), EVAL_LABELS)):
            plot_roc_curves(axes_nrd[0, j], nn_scores_data, idx, lbl)
            plot_det_curves(axes_nrd[1, j], nn_scores_data, idx, lbl)
        fig_nrd.tight_layout(rect=[0, 0, 1, 0.95])
        nrd_path = graphs_dir / "nn_roc_det.svg"
        fig_nrd.savefig(nrd_path, dpi=150, bbox_inches="tight")
        plt.close(fig_nrd)
        print(f"Saved → {nrd_path}")

    # Combined
    fig = plt.figure(figsize=(20, 19))
    fig.suptitle("Model Evaluation Results", fontsize=14, fontweight="bold")
    gs = GridSpec(4, 20, figure=fig, hspace=0.55, wspace=1.2, height_ratios=[0.6, 1.0, 1.0, 1.0])

    plot_metric_table(fig.add_subplot(gs[0, :]), data)

    plot_macro_f1(fig.add_subplot(gs[1, :10]), data)
    plot_pr_scatter(fig.add_subplot(gs[1, 10:15]), data)
    plot_inference_time(fig.add_subplot(gs[1, 15:]), data)

    n_models = len(data)
    cm_width = 20 // n_models
    for i, model in enumerate(data):
        lo = i * cm_width
        hi = (i + 1) * cm_width if i < n_models - 1 else 20
        plot_confusion_matrix(
            fig.add_subplot(gs[2, lo:hi]),
            data[model]["cm"],
            cm_classes,
            f"{MODEL_SHORT[model]} — Confusion Matrix",
        )

    plot_subclass_group(fig.add_subplot(gs[3, :10]), data, SPEECH_SUBS, SPEECH_SHORT, "Speech Subclasses — F1 by Model")
    plot_subclass_group(fig.add_subplot(gs[3, 10:]), data, MUSIC_SUBS, MUSIC_SHORT, "Music Subclasses — F1 by Model")

    out = RESULTS_DIR / "results.svg"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
