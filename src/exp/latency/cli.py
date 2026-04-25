"""CPU streaming latency comparison across all 6 classifiers.

Run with ``uv run smclassifier exp latency run`` from the project root.
Outputs ``src/exp/latency/results/latency_cpu.{md,svg}``.

CPU-only by design: GTX 1050 (sm_61) isn't supported by the current PyTorch
wheel. ``CUDA_VISIBLE_DEVICES=""`` is set before torch is imported anywhere
so torch never spins up CUDA libs.
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import click
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tomli

from ...demo.runner import RUNNERS

_CFG_PATH = Path(__file__).parent / "config.toml"
_RESULTS_DIR = Path(__file__).parent / "results"

# Plot styling — duplicated from scripts/visualize_results.py so this module
# stays importable without the scripts/ sys.path shim.
_COLORS = {
    "tcn_lstm": "#E91E63",
    "small_tcn": "#009688",
    "tcn": "#2196F3",
    "decision_tree": "#4CAF50",
    "svm": "#FF9800",
    "gmm": "#9C27B0",
}
_MODEL_SHORT = {
    "tcn_lstm": "TCN+LSTM",
    "small_tcn": "SmallTCN",
    "tcn": "TCN",
    "decision_tree": "DT",
    "svm": "SVM",
    "gmm": "GMM",
}


def _load_config() -> dict:
    with open(_CFG_PATH, "rb") as f:
        return tomli.load(f)["latency"]


def _prewarm_torch(num_threads: int) -> None:
    # Pay the one-time torch + safetensors import cost once so per-NN-model
    # load_ms reflects weight load, not "first NN bears the import". Pin
    # intra-op threads here while we're at it — set_num_threads must be
    # called before any parallel work, which the runner construction does.
    import torch
    from safetensors.torch import load_file  # noqa: F401
    torch.set_num_threads(num_threads)


def _deep_cleanup(passes: int, sleep_s: float) -> None:
    for _ in range(passes):
        gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    time.sleep(sleep_s)


def _bench_one(name: str, cfg: dict) -> dict:
    proc = psutil.Process()
    rss_pre = proc.memory_info().rss

    t = time.perf_counter()
    runner = RUNNERS[name]()
    load_ms = (time.perf_counter() - t) * 1000.0

    chunk_samples = runner.chunk_samples
    sr = runner.sr
    chunk_ms = chunk_samples / sr * 1000.0

    rng = np.random.default_rng(cfg["seed"])
    n_total = cfg["warmup_pushes"] + cfg["trials"] * cfg["pushes_per_trial"]
    inputs = (rng.standard_normal((n_total, chunk_samples)) * cfg["input_scale"]).astype(np.float32)

    for i in range(cfg["warmup_pushes"]):
        runner.push(inputs[i])

    all_ns: list[int] = []
    trial_medians: list[float] = []
    peak_abs = proc.memory_info().rss
    enabled = gc.isenabled()
    for k in range(cfg["trials"]):
        gc.disable()
        try:
            buf_ns = [0] * cfg["pushes_per_trial"]
            base = cfg["warmup_pushes"] + k * cfg["pushes_per_trial"]
            for i in range(cfg["pushes_per_trial"]):
                buf = inputs[base + i]
                t0 = time.perf_counter_ns()
                runner.push(buf)
                buf_ns[i] = time.perf_counter_ns() - t0
                r = proc.memory_info().rss
                if r > peak_abs:
                    peak_abs = r
        finally:
            if enabled:
                gc.enable()
        all_ns.extend(buf_ns)
        trial_medians.append(statistics.median(ns / 1e6 for ns in buf_ns))
        gc.collect()

    push_ms = sorted(ns / 1e6 for ns in all_ns)
    median = push_ms[len(push_ms) // 2]
    p99 = push_ms[int(0.99 * len(push_ms)) - 1]
    ms_per_sec = median / chunk_ms * 1000.0
    cv_pct = (
        statistics.stdev(trial_medians) / statistics.mean(trial_medians) * 100.0
        if len(trial_medians) > 1
        else 0.0
    )

    runner.close()
    del runner
    _deep_cleanup(cfg["cleanup_gc_passes"], cfg["cleanup_sleep_s"])
    rss_after = proc.memory_info().rss

    return {
        "name": name,
        "load_ms": load_ms,
        "chunk_ms": chunk_ms,
        "median_push_ms": median,
        "p99_push_ms": p99,
        "trial_median_cv_pct": cv_pct,
        "ms_per_sec_audio": ms_per_sec,
        "rtf": 1000.0 / ms_per_sec,
        "peak_rss_abs_mb": peak_abs / 1024**2,
        "peak_rss_marginal_mb": (peak_abs - rss_pre) / 1024**2,
        "rss_residual_mb": (rss_after - rss_pre) / 1024**2,
    }


def _cpu_label() -> str:
    try:
        with open("/proc/cpuinfo", "rt") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "n/a"


def _format_table(rows: list[dict], cfg: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_pooled = cfg["trials"] * cfg["pushes_per_trial"]
    max_abs = max(r["peak_rss_abs_mb"] for r in rows)
    header = (
        f"# Streaming inference latency — CPU only\n\n"
        f"- date: {now}\n"
        f"- cpu: {_cpu_label()}\n"
        f"- python: {sys.version.split()[0]}\n"
        f"- torch: {_torch_version()} (cpu-only forced via `CUDA_VISIBLE_DEVICES=\"\"`)\n"
        f"- torch threads: {cfg['torch_threads']} (intra-op, applied via `torch.set_num_threads`)\n"
        f"- warmup pushes: {cfg['warmup_pushes']}, "
        f"trials: {cfg['trials']} × {cfg['pushes_per_trial']} timed pushes "
        f"({n_pooled} pooled per model)\n"
        f"- input: random Gaussian noise scaled to ±{cfg['input_scale']}, seed {cfg['seed']}\n"
        f"- gc disabled inside the timed loop; `gc.collect()` between trials\n"
        f"- between-model cleanup: {cfg['cleanup_gc_passes']}× `gc.collect()` "
        f"+ glibc `malloc_trim(0)` + {cfg['cleanup_sleep_s']} s sleep\n"
        f"- max absolute RSS observed across the whole run: **{max_abs:.0f} MB**\n\n"
    )

    cols = [
        ("model", lambda r: _MODEL_SHORT.get(r["name"], r["name"]), "left"),
        ("load_ms", lambda r: f"{r['load_ms']:.1f}", "right"),
        ("chunk_ms", lambda r: f"{r['chunk_ms']:.2f}", "right"),
        ("median_ms", lambda r: f"{r['median_push_ms']:.3f}", "right"),
        ("p99_ms", lambda r: f"{r['p99_push_ms']:.3f}", "right"),
        ("cv%", lambda r: f"{r['trial_median_cv_pct']:.1f}", "right"),
        ("ms/s_audio", lambda r: f"{r['ms_per_sec_audio']:.1f}", "right"),
        ("rtf", lambda r: f"{r['rtf']:.1f}×", "right"),
        ("peak_rss_mb", lambda r: f"{r['peak_rss_abs_mb']:.0f}", "right"),
        ("Δrss_mb", lambda r: f"{r['peak_rss_marginal_mb']:.0f}", "right"),
        ("residual_mb", lambda r: f"{r['rss_residual_mb']:.0f}", "right"),
    ]

    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("---:" if c[2] == "right" else "---" for c in cols) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(c[1](r) for c in cols) + " |")
    table = "\n".join(lines)

    footer = (
        "\n\n**ms/s_audio** is the fair cross-model metric: each model's chunk\n"
        "covers a different audio duration (DT 10 ms, GMM/SVM 15 ms, NN family\n"
        "23.2 ms), so raw `median_ms` is **not** comparable. **rtf > 1×** means\n"
        "faster than real time; rtf < 1× cannot keep up with a live mic.\n\n"
        "**cv%** is the coefficient of variation across the trial medians —\n"
        "a quick reliability check. Numbers under ~10% mean the median is\n"
        "stable; higher suggests background load on the machine.\n\n"
        "**peak_rss_mb** is the absolute peak RSS during the model's run, but\n"
        "is **run-order-dependent** — a model that ran after a heavy cleanup\n"
        "starts from a lower baseline, so its absolute peak under-reports its\n"
        "true standalone footprint. **Δrss_mb** is the more stable per-model\n"
        "metric: the marginal cost on top of whatever baseline was present.\n"
        "**residual_mb** is what survived the deep cleanup — modules and\n"
        "pickle pools that don't get released between models. The header line\n"
        "above (max absolute RSS observed) reflects the realistic ceiling\n"
        "for a long-running session that visits every model.\n\n"
        "Caveats: random noise undertests sklearn DT/SVM data-dependent paths;\n"
        "the bias is expected to be small but not zero. Numbers reflect this\n"
        "machine + this PyTorch build only.\n"
    )

    return header + table + footer


def _plot(rows: list[dict], out_svg: Path) -> None:
    by_speed = sorted(rows, key=lambda r: r["ms_per_sec_audio"])
    names = [r["name"] for r in by_speed]
    labels = [_MODEL_SHORT.get(n, n) for n in names]
    colors = [_COLORS.get(n, "#888888") for n in names]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel A — ms per second of audio (lower is better). Log-x because RTF
    # spans ~3 orders of magnitude across the model family.
    vals = [r["ms_per_sec_audio"] for r in by_speed]
    ax_a.barh(labels, vals, color=colors)
    ax_a.set_xscale("log")
    ax_a.axvline(1000.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                 label="real-time threshold")
    ax_a.set_xlabel("ms wall-clock per second of audio (log scale)")
    ax_a.set_title("Streaming latency (lower is better)")
    ax_a.legend(loc="lower right", fontsize=9)
    for i, r in enumerate(by_speed):
        ax_a.text(vals[i] * 1.05, i, f"{r['rtf']:.1f}× RT",
                  va="center", fontsize=9)
    ax_a.invert_yaxis()

    # Panel B — marginal RSS (per-model cost on top of pre-load baseline).
    # Absolute peak is run-order-dependent due to imperfect between-model
    # cleanup, so the marginal number is the stable comparator.
    rss_vals = [r["peak_rss_marginal_mb"] for r in by_speed]
    ax_b.barh(labels, rss_vals, color=colors)
    ax_b.set_xlabel("marginal RSS — model cost on top of baseline (MB)")
    ax_b.set_title("Memory footprint (Δ vs pre-load)")
    rss_max = max(rss_vals) if rss_vals else 1.0
    for i, v in enumerate(rss_vals):
        ax_b.text(v + rss_max * 0.01, i, f"{v:.0f} MB",
                  va="center", fontsize=9)
    ax_b.invert_yaxis()

    fig.suptitle(f"CPU streaming latency — {_cpu_label()}", fontsize=11)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    plt.close(fig)


@click.group("latency")
def latency_group():
    """CPU streaming latency benchmark across all 6 classifiers."""


@latency_group.command("run")
def run():
    """Bench all models; write results/latency_cpu_t{N}.{md,svg}."""
    cfg = _load_config()
    n_threads = cfg["torch_threads"]
    out_md = _RESULTS_DIR / f"latency_cpu_t{n_threads}.md"
    out_svg = _RESULTS_DIR / f"latency_cpu_t{n_threads}.svg"

    _prewarm_torch(n_threads)
    rows: list[dict] = []
    for name in cfg["models"]:
        click.echo(f"benchmarking {name} ...")
        rows.append(_bench_one(name, cfg))

    table = _format_table(rows, cfg)
    click.echo()
    click.echo(table)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table)
    _plot(rows, out_svg)
    click.echo(f"\nsaved → {out_md}")
    click.echo(f"saved → {out_svg}")
