"""Visualize evaluation results for all models."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["tcn", "decision_tree", "svm", "gmm"]
COLORS = {"tcn": "#2196F3", "decision_tree": "#4CAF50", "svm": "#FF9800", "gmm": "#9C27B0"}


def parse_eval(path: Path) -> dict:
    text = path.read_text()

    def macro_f1(section: str) -> float | None:
        m = re.search(r"Macro\s+([\d.]+)", section)
        return float(m.group(1)) if m else None

    two_class = re.search(r"── 2-class.*?── 3-class", text, re.DOTALL)
    three_class = re.search(r"── 3-class.*?── By subclass", text, re.DOTALL)
    subclass_section = re.search(r"── By subclass.*", text, re.DOTALL)

    subclasses = {}
    if subclass_section:
        for line in subclass_section.group().splitlines()[1:]:
            m = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                subclasses[m.group(1)] = {
                    "f1": float(m.group(2)),
                    "p": float(m.group(3)),
                    "r": float(m.group(4)),
                }

    return {
        "2class_macro_f1": macro_f1(two_class.group()) if two_class else None,
        "3class_macro_f1": macro_f1(three_class.group()) if three_class else None,
        "subclasses": subclasses,
    }


def main():
    data = {}
    for model in MODELS:
        path = RESULTS_DIR / f"{model}.eval"
        if path.exists():
            data[model] = parse_eval(path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Evaluation Results", fontsize=14, fontweight="bold")

    # --- Plot 1: 2-class vs 3-class macro F1 ---
    ax = axes[0]
    x = np.arange(len(data))
    w = 0.35
    labels = list(data.keys())
    f1_2 = [data[m]["2class_macro_f1"] or 0 for m in labels]
    f1_3 = [data[m]["3class_macro_f1"] or 0 for m in labels]

    bars2 = ax.bar(x - w / 2, f1_2, w, label="2-class", color=[COLORS[m] for m in labels], alpha=0.9)
    bars3 = ax.bar(x + w / 2, f1_3, w, label="3-class", color=[COLORS[m] for m in labels], alpha=0.5, hatch="//")

    ax.axhline(0.88, color="red", linestyle="--", linewidth=1, label="2-class target (0.88)")
    ax.axhline(0.85, color="orange", linestyle="--", linewidth=1, label="3-class target (0.85)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in labels])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Macro F1 by Eval Mode")
    ax.set_ylabel("Macro F1")
    ax.legend(fontsize=7)

    for bar in [*bars2, *bars3]:
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    # --- Plot 2: Per-subclass F1 for TCN ---
    ax = axes[1]
    tcn_sub = data.get("tcn", {}).get("subclasses", {})
    if tcn_sub:
        names = list(tcn_sub.keys())
        f1s = [tcn_sub[n]["f1"] for n in names]
        short = [n.replace("speech_", "sp/").replace("music_", "mu/").replace("_over_music", "+mu") for n in names]
        bar_colors = ["#2196F3" if n.startswith("sp") else "#FF5722" if n.startswith("mu/") else "#607D8B" for n in short]
        bars = ax.barh(short, f1s, color=bar_colors, alpha=0.85)
        ax.axvline(0.9, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.05)
        ax.set_title("TCN — Per-subclass F1")
        ax.set_xlabel("F1")
        for bar, val in zip(bars, f1s):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7)

    # --- Plot 3: Subclass F1 comparison across models (speech/music only) ---
    ax = axes[2]
    all_subclasses = sorted({s for m in data.values() for s in m["subclasses"]})
    x = np.arange(len(all_subclasses))
    width = 0.2
    for i, model in enumerate(data):
        vals = [data[model]["subclasses"].get(s, {}).get("f1", 0) for s in all_subclasses]
        ax.bar(x + i * width, vals, width, label=model, color=COLORS[model], alpha=0.85)

    short_names = [s.replace("speech_", "sp/").replace("music_", "mu/").replace("_over_music", "+mu") for s in all_subclasses]
    ax.set_xticks(x + width * (len(data) - 1) / 2)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-subclass F1 — All Models")
    ax.set_ylabel("F1")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = RESULTS_DIR / "results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
