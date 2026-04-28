"""CPU complexity analysis across all 6 classifiers.

Combines symbolic complexity (params, MACs/frame, receptive field, critical
path) with empirical streaming measurements (wall-clock latency, RTF, RSS)
and Pareto plots against F1_macro.

Run with ``uv run smclassifier exp complexity run`` from the project root.
Outputs ``src/exp/complexity/results/complexity_t{N}.{md,svg}``.
``uv run smclassifier exp complexity scale`` sweeps thread counts and fits
Amdahl per model to ``src/exp/complexity/results/complexity_scale.{md,svg}``.

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
        # Old [latency] section name kept for backward compatibility with the
        # existing config file; the values themselves are unchanged.
        data = tomli.load(f)
    return data.get("complexity", data.get("latency", {}))


# ---------------------------------------------------------------------------
# Symbolic complexity — derived once per runner from the loaded model.
# ---------------------------------------------------------------------------

# Result schema for `_symbolic_complexity`:
#   params         : trainable params + persistent buffers (count of floats)
#   macs_per_frame : MACs per output frame in streaming mode (one chunk in,
#                    one decision out). Lower is better.
#   recept_samples : receptive field in raw audio samples
#   recept_ms      : same, in milliseconds (sr-aware)
#   critical_path  : longest serial chain of ops per frame (T_inf)
#   ideal_par      : T1 / T_inf — algorithmic max parallelism

def _symbolic_nn(runner) -> dict:
    """Symbolic counts for a TCN-family runner (NNRunner)."""
    import torch
    model = runner._model
    tcn = model.model  # CausalTCN (with optional return_features)
    n_filters = tcn.n_filters
    blocks = list(tcn.tcn)
    n_blocks = len(blocks)
    # Kernel size is uniform across blocks in this codebase.
    kernel = blocks[0].conv1.conv.kernel_size[0]
    # n_input channels into the first conv (after input_proj 1x1).
    in_ch_proj = tcn.input_proj.in_channels  # = n_features (mel / Δ-Δ²)
    n_classes = tcn.classifier.out_channels

    # Receptive field in feature frames — read dilation directly from each
    # block (each block has 2 dilated convs at the same dilation, contributing
    # 2·(k−1)·dilation extra context). +1 for the centre frame.
    rf_frames = 1 + sum(
        2 * (kernel - 1) * blk.conv1.conv.dilation[0] for blk in blocks
    )
    # Map to audio samples + ms via the streaming hop.
    rf_samples = rf_frames * runner.chunk_samples
    rf_ms = rf_samples / runner.sr * 1000.0

    # Params + buffers.
    p_train = sum(p.numel() for p in model.parameters())
    p_buf = sum(b.numel() for b in model.buffers())

    # MACs per OUTPUT frame (offline, one frame).
    # Each TCN block: two k-tap convs at n_filters→n_filters → 2·k·n_filters^2.
    # Plus input_proj (1·n_features·n_filters) and classifier (1·n_filters·n_classes).
    macs_offline = (
        in_ch_proj * n_filters
        + n_blocks * 2 * kernel * n_filters * n_filters
        + n_filters * n_classes
    )

    # Optional preprocessor (e.g. conv1d block) before the TCN backbone.
    preproc = getattr(model, "preproc", None)
    pre_macs = 0
    if preproc is not None:
        for sub in preproc.modules():
            if isinstance(sub, torch.nn.Conv1d):
                k = sub.kernel_size[0]
                pre_macs += k * sub.in_channels * sub.out_channels

    # Optional tail (LSTM): one step per emitted frame on top of TCN features.
    tail = getattr(model, "tail", None)
    tail_macs = 0
    tail_state_floats = 0
    if tail is not None:
        for sub in tail.modules():
            if isinstance(sub, torch.nn.LSTM):
                H = sub.hidden_size
                D = sub.input_size
                tail_macs += 4 * H * (D + H)
                tail_state_floats += 2 * H  # (h, c)
            elif isinstance(sub, torch.nn.GRU):
                H = sub.hidden_size
                D = sub.input_size
                tail_macs += 3 * H * (D + H)
                tail_state_floats += H

    macs_offline += pre_macs

    # Streaming MACs/frame = cost of one push().
    # Current StreamingInference reruns the TCN over the whole left-RF buffer
    # each call (≈ rf_frames feature frames) since the TCN backbone is
    # stateless; the LSTM tail (if any) only steps on the *new* frame.
    macs_streaming = macs_offline * rf_frames + tail_macs

    # Critical path (T_inf): conv layers form the serial chain. Each block is
    # 2 layers deep; plus input_proj and classifier add 2; +1 for tail step.
    critical_path = 2 + 2 * n_blocks
    if tail_macs > 0:
        critical_path += 1

    # Persistent streaming state in floats: audio-sample ring buffer + tail h/c.
    state_floats = rf_samples + tail_state_floats
    # Buffer in ms = audio-equivalent of all persistent state. The audio ring
    # buffer dominates (rf_samples is a sample count, h/c is tens of floats),
    # so this is essentially the audio context held between calls.
    state_ms = state_floats / runner.sr * 1000.0

    return {
        "params": p_train,
        "buffers": p_buf,
        "params_total": p_train + p_buf,
        "macs_offline": macs_offline,
        "macs_per_frame": macs_streaming,
        "recept_samples": rf_samples,
        "recept_ms": rf_ms,
        "state_floats": state_floats,
        "state_ms": state_ms,
        "critical_path": critical_path,
        "ideal_par": macs_streaming / max(1, critical_path),
    }


def _symbolic_classic(runner) -> dict:
    """Symbolic counts for a sklearn-backed StreamingClassifier."""
    streaming = runner._cls
    cls = streaming.model  # DecisionTree / SVM / GMM wrapper
    fe = streaming.fe       # FeatExtractor — has lt_len for the audio context

    fh = runner.chunk_samples
    sr = runner.sr

    macs_per_frame = 0
    params = 0
    state_floats = 0
    critical_path = 1

    cls_name = cls.__class__.__name__
    n_classes = 3

    if cls_name == "DecisionTree":
        clf = cls.tree.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "tree_"):
            tree = clf.tree_
            depth = int(tree.max_depth)
            params = int(tree.node_count)
            # One feature compare per depth level on the inference path.
            macs_per_frame = depth
            critical_path = depth
        # Smoothing — exponentially weighted sum over last decisions.
        n_smooth = (cls.last_decisions.maxlen or 0)
        state_floats += n_smooth * n_classes
        # Weighted sum: n_smooth · n_classes MACs.
        macs_per_frame += n_smooth * n_classes

    elif cls_name == "SVM":
        svc = cls.svm.named_steps.get("svm")
        if svc is not None and hasattr(svc, "support_vectors_"):
            n_sv = int(svc.support_vectors_.shape[0])
            D = int(svc.support_vectors_.shape[1])
            macs_per_frame = n_sv * D
            params = n_sv * D + n_sv + 1
            critical_path = max(1, int(np.ceil(np.log2(max(2, n_sv)))))
        # Smoothing — mean over last `dec_buf.maxlen` decision vectors.
        n_smooth = (cls.dec_buf.maxlen or 0)
        state_floats += n_smooth * n_classes

    elif cls_name == "GMM":
        # Three GMMs (speech, music, inactive); sum their costs.
        K = D = 0
        for attr in ("gmm_speech", "gmm_music", "gmm_inactive"):
            g = getattr(cls, attr, None)
            if g is None or not hasattr(g, "means_"):
                continue
            K = g.n_components
            D = g.means_.shape[1]
            macs_per_frame += K * D       # log-likelihood over K Gaussians
            params += K * (2 * D + 1)     # diag-cov: μ + σ² + π per comp
        critical_path = max(1, int(np.ceil(np.log2(max(2, K))))) if K else 1
        # Scaler params (RobustScaler keeps median + IQR per feature).
        scaler = getattr(cls, "scaler", None)
        if scaler is not None and hasattr(scaler, "center_"):
            params += 2 * scaler.center_.size
        # Smoothing — mean over last `ll_buf` log-likelihood deltas.
        n_smooth = (cls.ll_buf.maxlen or 0)
        state_floats += n_smooth * n_classes

    # Receptive field is the FeatExtractor's signal context (lt_len_ms in the
    # config: 300 ms for DT, 1000 ms for GMM/SVM). The classifier itself sees
    # one frame, but the features were computed over this longer window.
    rf_samples = max(fh, getattr(fe, "lt_len", fh))
    state_floats += rf_samples  # signal_buffer is part of streaming state

    return {
        "params": params,
        "buffers": 0,
        "params_total": params,
        "macs_offline": macs_per_frame,
        "macs_per_frame": macs_per_frame,
        "recept_samples": rf_samples,
        "recept_ms": rf_samples / sr * 1000.0,
        "state_floats": state_floats,
        "state_ms": state_floats / sr * 1000.0,
        "critical_path": critical_path,
        "ideal_par": macs_per_frame / max(1, critical_path),
    }


def _symbolic_complexity(name: str, runner) -> dict:
    """Dispatch to the per-family symbolic-complexity helper."""
    if name in ("decision_tree", "gmm", "svm"):
        return _symbolic_classic(runner)
    return _symbolic_nn(runner)


# ---------------------------------------------------------------------------
# F1_macro from results/*.eval — joined into the per-model row for Pareto.
# ---------------------------------------------------------------------------

_RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results"


def _read_f1_macro(model_name: str) -> float | None:
    """Best-effort F1_macro extraction from ``results/<model>.eval``.

    Returns ``None`` if the file is missing or unparsable; callers gate the
    Pareto plot on this so an absent file isn't fatal.
    """
    p = _RESULTS_ROOT / f"{model_name}.eval"
    if not p.exists():
        return None
    try:
        text = p.read_text()
    except OSError:
        return None
    # The eval files are produced by `format_report` in src/evaluator.py and
    # contain a line like "Macro F1 (3-class):  0.9033". Match loosely so
    # minor format drift doesn't break the join.
    import re
    m = re.search(r"Macro F1[^0-9]+([0-9]+\.[0-9]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


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

    # Symbolic complexity is independent of the timing run but cheap to derive
    # while we have the runner; doing it here keeps everything in one row.
    try:
        sym = _symbolic_complexity(name, runner)
    except Exception as exc:
        sym = {"error": str(exc)}

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
        "f1_macro": _read_f1_macro(name),
        "symbolic": sym,
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


def _fmt_macs(v: int) -> str:
    """Compact MAC formatting (M / G / T)."""
    if v <= 0:
        return "—"
    if v >= 1e9:
        return f"{v/1e9:.2f} G"
    if v >= 1e6:
        return f"{v/1e6:.2f} M"
    if v >= 1e3:
        return f"{v/1e3:.1f} k"
    return str(v)


def _fmt_params(v: int) -> str:
    """Compact param formatting (K / M)."""
    if v <= 0:
        return "—"
    if v >= 1e6:
        return f"{v/1e6:.2f} M"
    if v >= 1e3:
        return f"{v/1e3:.1f} k"
    return str(v)


def _format_table(rows: list[dict], cfg: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_pooled = cfg["trials"] * cfg["pushes_per_trial"]
    max_abs = max(r["peak_rss_abs_mb"] for r in rows)
    header = (
        f"# CPU complexity analysis — front-of-pipeline classifiers\n\n"
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

    # Empirical (timing + memory) table.
    emp_cols = [
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
        ("F1_macro", lambda r: f"{r['f1_macro']:.4f}" if r.get("f1_macro") is not None else "—", "right"),
    ]

    head = "| " + " | ".join(c[0] for c in emp_cols) + " |"
    sep = "|" + "|".join("---:" if c[2] == "right" else "---" for c in emp_cols) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(c[1](r) for c in emp_cols) + " |")
    emp_table = "\n".join(lines)

    # Symbolic complexity table (closed-form from architecture).
    def _sym(r, k, default="—"):
        return r.get("symbolic", {}).get(k, default)

    sym_cols = [
        ("model", lambda r: _MODEL_SHORT.get(r["name"], r["name"]), "left"),
        ("params", lambda r: _fmt_params(_sym(r, "params", 0)), "right"),
        ("MACs/frame", lambda r: _fmt_macs(_sym(r, "macs_per_frame", 0)), "right"),
        ("RF_ms", lambda r: f"{_sym(r, 'recept_ms', 0):.1f}" if isinstance(_sym(r, "recept_ms"), (int, float)) else "—", "right"),
        ("buf_ms", lambda r: f"{_sym(r, 'state_ms', 0):.1f}" if isinstance(_sym(r, "state_ms"), (int, float)) else "—", "right"),
        ("T∞", lambda r: f"{_sym(r, 'critical_path', 0)}", "right"),
        ("T₁/T∞", lambda r: f"{_sym(r, 'ideal_par', 0):.0f}" if isinstance(_sym(r, "ideal_par"), (int, float)) else "—", "right"),
    ]
    head2 = "| " + " | ".join(c[0] for c in sym_cols) + " |"
    sep2 = "|" + "|".join("---:" if c[2] == "right" else "---" for c in sym_cols) + "|"
    lines2 = [head2, sep2]
    for r in rows:
        lines2.append("| " + " | ".join(c[1](r) for c in sym_cols) + " |")
    sym_table = "\n".join(lines2)

    footer = (
        "\n\n## Empirical (1-thread CPU)\n\n"
        + emp_table
        + "\n\n## Symbolic complexity (per output frame)\n\n"
        + sym_table
        + "\n\n**ms/s_audio** is the fair cross-model metric: each model's chunk\n"
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
        "**MACs/frame** is the streaming cost (one push). For the TCN family\n"
        "this includes the receptive-field re-run that the current\n"
        "`StreamingInference` performs; an idealised stateful cache would\n"
        "drop this by `recept_frames`× to the offline number.\n\n"
        "**RF_ms** is the audio context the model needs to compute one output\n"
        "frame; **buf_ms** is the audio-equivalent of all persistent streaming\n"
        "state (audio ring buffer + LSTM h/c + smoothing windows), divided by\n"
        "the model's own sample rate. They're nearly equal because the audio\n"
        "ring buffer dominates: the LSTM h/c (~128 floats) and the classic\n"
        "smoothing buffers (n_smooth × n_classes) add only a few ms each.\n\n"
        "**T∞** is the critical-path depth (longest serial chain of ops);\n"
        "**T₁/T∞ = MACs/frame ÷ T∞** is the algorithmic max parallelism.\n"
        "DT is dominated by tree-depth comparisons (intrinsically serial,\n"
        "T₁/T∞ ≈ 2 once smoothing is included); SVM is a parallel-friendly\n"
        "kernel sum; the TCN family scores 10⁵–10⁶ in theory but BLAS at\n"
        "batch=1 only realises a fraction of that — see the scale sweep.\n\n"
        "Caveats: random noise undertests sklearn DT/SVM data-dependent paths;\n"
        "the bias is expected to be small but not zero. Numbers reflect this\n"
        "machine + this PyTorch build only.\n"
    )

    return header + footer


def _plot(rows: list[dict], out_svg: Path) -> None:
    by_speed = sorted(rows, key=lambda r: r["ms_per_sec_audio"])
    labels = [_MODEL_SHORT.get(r["name"], r["name"]) for r in by_speed]
    colors = [_COLORS.get(r["name"], "#888888") for r in by_speed]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # Panel A — ms per second of audio (lower is better). Log-x because RTF
    # spans ~3 orders of magnitude across the model family.
    vals = [r["ms_per_sec_audio"] for r in by_speed]
    ax_a.barh(labels, vals, color=colors)
    ax_a.set_xscale("log")
    ax_a.axvline(1000.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                 label="real-time threshold")
    ax_a.set_xlabel("ms wall-clock per second of audio (log scale)")
    ax_a.set_title("Streaming wall-clock (lower is better)")
    ax_a.legend(loc="lower right", fontsize=9)
    for i, r in enumerate(by_speed):
        ax_a.text(vals[i] * 1.05, i, f"{r['rtf']:.1f}× RT",
                  va="center", fontsize=9)
    ax_a.invert_yaxis()

    # Panel B — symbolic MACs per frame (architectural compute cost).
    macs_vals = [max(1, r.get("symbolic", {}).get("macs_per_frame", 1)) for r in by_speed]
    ax_b.barh(labels, macs_vals, color=colors)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("MACs per output frame (streaming, log scale)")
    ax_b.set_title("Symbolic compute (lower is cheaper)")
    for i, v in enumerate(macs_vals):
        ax_b.text(v * 1.1, i, _fmt_macs(int(v)), va="center", fontsize=9)
    ax_b.invert_yaxis()

    # Panel C — marginal RSS (per-model cost on top of pre-load baseline).
    rss_vals = [r["peak_rss_marginal_mb"] for r in by_speed]
    ax_c.barh(labels, rss_vals, color=colors)
    ax_c.set_xlabel("marginal RSS — model cost on top of baseline (MB)")
    ax_c.set_title("Memory footprint (Δ vs pre-load)")
    rss_max = max(rss_vals) if rss_vals else 1.0
    for i, v in enumerate(rss_vals):
        ax_c.text(v + rss_max * 0.01, i, f"{v:.0f} MB",
                  va="center", fontsize=9)
    ax_c.invert_yaxis()

    # Panel D — Pareto: F1_macro vs ms/s_audio. Models without an .eval
    # file are dropped; the rest get a labelled scatter point.
    pareto = [r for r in rows if r.get("f1_macro") is not None]
    if pareto:
        xs = [r["ms_per_sec_audio"] for r in pareto]
        ys = [r["f1_macro"] for r in pareto]
        cs = [_COLORS.get(r["name"], "#888888") for r in pareto]
        ax_d.scatter(xs, ys, c=cs, s=80, edgecolor="black", linewidth=0.5, zorder=3)
        for r in pareto:
            ax_d.annotate(
                _MODEL_SHORT.get(r["name"], r["name"]),
                xy=(r["ms_per_sec_audio"], r["f1_macro"]),
                xytext=(6, 4), textcoords="offset points", fontsize=9,
            )
        ax_d.set_xscale("log")
        ax_d.axvline(1000.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_d.set_xlabel("ms wall-clock per second of audio (log scale)")
        ax_d.set_ylabel("F1_macro (3-class)")
        ax_d.set_title("Quality vs cost — Pareto")
        ax_d.grid(alpha=0.3, zorder=0)
    else:
        ax_d.text(0.5, 0.5, "no F1 data — run `eval` for each model first",
                  ha="center", va="center", transform=ax_d.transAxes,
                  fontsize=10, color="gray")
        ax_d.set_axis_off()

    fig.suptitle(f"CPU complexity — {_cpu_label()}", fontsize=12)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Thread-scaling sweep — multi-N timing + Amdahl fit.
# ---------------------------------------------------------------------------


def _amdahl_fit(thread_counts: list[int], speedups: list[float]) -> float:
    """Least-squares fit of Amdahl's `p` to (N, S(N)) pairs.

    `S(N) = 1 / ((1-p) + p/N)` ⇒ for each (N, S), p = (1 - 1/S) · N / (N - 1).
    We average those per-N estimates (skipping N=1) — robust enough for the
    handful of points we have, no scipy needed.

    Returned `p` is **not** clamped to [0, 1]: a negative value signals
    anti-scaling (S(N) < 1, threading overhead dominates) and we want that
    visible in the report rather than collapsed to 0.
    """
    ps: list[float] = []
    for n, s in zip(thread_counts, speedups):
        if n == 1 or s <= 0:
            continue
        ps.append((1.0 - 1.0 / s) * (n / (n - 1)))
    return sum(ps) / len(ps) if ps else 0.0


def _format_scale_table(per_n: dict[int, list[dict]]) -> str:
    """Markdown table: rows = models, columns = thread counts (median ms),
    plus speedups + Amdahl `p`."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    Ns = sorted(per_n.keys())
    # Models in the order of the smallest N's run.
    model_order = [r["name"] for r in per_n[Ns[0]]]

    header = (
        f"# CPU thread-scaling sweep\n\n"
        f"- date: {now}\n"
        f"- cpu: {_cpu_label()}\n"
        f"- thread counts: {Ns}\n\n"
    )

    cols = [("model", "left")] + [(f"t{n}_ms", "right") for n in Ns]
    cols += [(f"S({Ns[-1]})", "right"), ("amdahl_p", "right")]
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("---:" if c[1] == "right" else "---" for c in cols) + "|"
    lines = [head, sep]
    for name in model_order:
        row_vals = []
        n1_med = None
        speedups = []
        for n in Ns:
            r = next((rr for rr in per_n[n] if rr["name"] == name), None)
            med = r["median_push_ms"] if r else float("nan")
            row_vals.append(f"{med:.3f}")
            if n == 1:
                n1_med = med
            if n1_med and med > 0:
                speedups.append(n1_med / med)
            else:
                speedups.append(1.0)
        s_max = speedups[-1] if speedups else 1.0
        amdahl_p = _amdahl_fit(Ns, speedups)
        lines.append(
            "| " + _MODEL_SHORT.get(name, name) + " | "
            + " | ".join(row_vals)
            + f" | {s_max:.2f}× | {amdahl_p:.2f} |"
        )

    footer = (
        "\n\n**S(N)** = t_f(1) / t_f(N) — observed parallel speedup at the\n"
        "highest thread count tested. **amdahl_p** is the parallel fraction\n"
        "fitted to all (N, S(N)) pairs: ≈0 means single-threaded by\n"
        "construction, ≈1 means linear scaling, **negative** means anti-\n"
        "scaling (S(N)<1 — threading overhead beats any parallel gain, as\n"
        "happens with sklearn classifiers that aren't BLAS-bound during\n"
        "predict). For a front-of-pipeline gate the user typically gives\n"
        "the model 1 core, so multi-thread speedup is informational rather\n"
        "than load-bearing.\n"
    )
    return header + "\n".join(lines) + footer


def _plot_scale(per_n: dict[int, list[dict]], out_svg: Path) -> None:
    Ns = sorted(per_n.keys())
    model_order = [r["name"] for r in per_n[Ns[0]]]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in model_order:
        meds = []
        for n in Ns:
            r = next((rr for rr in per_n[n] if rr["name"] == name), None)
            meds.append(r["median_push_ms"] if r else float("nan"))
        n1 = meds[0] if meds and meds[0] else 1.0
        speedups = [n1 / m if m else float("nan") for m in meds]
        ax.plot(Ns, speedups, marker="o",
                color=_COLORS.get(name, "#888888"),
                label=_MODEL_SHORT.get(name, name))
    ax.plot(Ns, Ns, linestyle="--", color="gray", alpha=0.5, label="ideal (linear)")
    ax.set_xlabel("torch threads N")
    ax.set_ylabel("speedup S(N) = t(1) / t(N)")
    ax.set_title(f"Thread scaling — {_cpu_label()}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("complexity")
def complexity_group():
    """CPU complexity analysis (symbolic + empirical) across all 6 classifiers."""


@complexity_group.command("run")
def run():
    """Bench all models; write results/complexity_t{N}.{md,svg}."""
    cfg = _load_config()
    n_threads = cfg["torch_threads"]
    out_md = _RESULTS_DIR / f"complexity_t{n_threads}.md"
    out_svg = _RESULTS_DIR / f"complexity_t{n_threads}.svg"

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


@complexity_group.command("scale")
@click.option("--threads", "-t", multiple=True, type=int,
              help="Thread counts to sweep (default: 1,2,4,n_cores).")
def scale(threads):
    """Sweep torch_threads, fit Amdahl `p` per model."""
    cfg = _load_config()
    if not threads:
        # Sweep up to logical cores so SMT/hyperthreading shows as a separate
        # data point — physical-cores alone often hides the diminishing-returns
        # knee where threading stops helping.
        n_phys = psutil.cpu_count(logical=False) or 1
        n_log = psutil.cpu_count(logical=True) or n_phys
        Ns = sorted(set([1, 2, n_phys, n_log]))
    else:
        Ns = sorted(set(threads))
    click.echo(f"thread sweep: {Ns}")

    per_n: dict[int, list[dict]] = {}
    # NOTE: torch is imported once; set_num_threads can be re-applied
    # mid-process. The bench harness already uses `torch.set_num_threads`
    # in `_prewarm_torch`, but we call it here once per N to pin before
    # each pass.
    import torch
    _prewarm_torch(Ns[0])
    for n in Ns:
        click.echo(f"\n=== threads = {n} ===")
        torch.set_num_threads(n)
        rows: list[dict] = []
        for name in cfg["models"]:
            click.echo(f"  benchmarking {name} ...")
            rows.append(_bench_one(name, cfg))
        per_n[n] = rows

    table = _format_scale_table(per_n)
    click.echo()
    click.echo(table)
    out_md = _RESULTS_DIR / "complexity_scale.md"
    out_svg = _RESULTS_DIR / "complexity_scale.svg"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table)
    _plot_scale(per_n, out_svg)
    click.echo(f"\nsaved → {out_md}")
    click.echo(f"saved → {out_svg}")
