# Preprocessor sweep — label / param-count audit

Audit triggered by the question: in `fig:preprocessor_results`, is the `1 × 1` conv really `~57×` bigger than the `3 × 3` conv?

Short answer: **the param counts are correct, but the `1 × 1` / `3 × 3` framing is misleading**. The disparity comes from channel dimensions, not kernel size.

## Verified parameter counts

Computed directly from `src/nn/preprocessors.py` with `n_features=80` (the 80-mel baseline frontend used in this sweep):

| Preprocessor | Total trainable params | Breakdown |
|---|---:|---|
| `Conv1dPreprocessor` | **38,880** | 2 × `CausalConv1d(80, 80, kernel=3)` + 2 × `BatchNorm1d(80)` |
| `Conv2dPreprocessor` | **675** | `Conv2d(1, 32, 3×3)` + `Conv2d(32, 1, 3×3)` + 2 × `BatchNorm2d` |

Per-tensor breakdown:

```
Conv1dPreprocessor (38,880 params):
  conv1.weight  (80, 80, 3)  = 19,200
  conv1.bias    (80,)        =      80
  bn1.weight    (80,)        =      80
  bn1.bias      (80,)        =      80
  conv2.weight  (80, 80, 3)  = 19,200
  conv2.bias    (80,)        =      80
  bn2.weight    (80,)        =      80
  bn2.bias      (80,)        =      80

Conv2dPreprocessor (675 params):
  conv1.weight  (32, 1, 3, 3) = 288
  conv1.bias    (32,)         =  32
  bn1.weight    (32,)         =  32
  bn1.bias      (32,)         =  32
  conv2.weight  (1, 32, 3, 3) = 288
  conv2.bias    (1,)          =   1
  bn2.weight    (1,)          =   1
  bn2.bias      (1,)          =   1
```

Reproduce:

```bash
uv run python -c "
from src.nn.preprocessors import Conv1dPreprocessor, Conv2dPreprocessor
for name, cls in [('conv1d', Conv1dPreprocessor), ('conv2d', Conv2dPreprocessor)]:
    m = cls(80)
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'{name}: {total:,} params')
"
```

## Why the gap is so wide

Both preprocessors use a **kernel of size 3** along time. Neither is a `1 × 1` conv. The ~57× param-count disparity is driven entirely by **channel counts**:

| Preprocessor | Per-layer weight tensor | Layers | Per-layer weights | Notes |
|---|---|---:|---:|---|
| `Conv1dPreprocessor` | `(out=80, in=80, kT=3)` | 2 | 19,200 | 80→80 channel-mixing matrix dominates |
| `Conv2dPreprocessor` | `(out, in, kF=3, kT=3)` with channels `1→32→1` | 2 | 288 each | Tiny because of 1→32 / 32→1 channel reduction |

If `Conv1dPreprocessor` were a true `1 × 1` (kernel-1) conv, it would still have 80 × 80 × 1 × 2 = 12,800 weights — still ~19× larger than the 2-D variant, again because of channel dimensions.

Kernel area is a minor factor; the dominant axis is `(C_in × C_out)`.

## Label / prose mismatch

Two places in the thesis use the `1 × 1` framing:

1. **Figure y-axis label** in `fig:preprocessor_results` reads `1 × 1 conv` and `3 × 3 conv`.
2. **Chapter prose** in §6.2.3 (`subsec:exp_preprocessor`) currently reads:

   > Two variants are tested: a 1-D ($1 \times 1$) conv that mixes mel bins at each time step, and a 2-D conv that mixes both axes.

The implementation in `src/nn/preprocessors.py` contradicts both:

- `Conv1dPreprocessor` uses `kernel_size=3` in time. Each output frame mixes the previous **three** frames and all 80 mel bins, not "each time step" only.
- `Conv2dPreprocessor` is a 3 × 3 conv but operates on a `(B, 1, F, T)` tensor with mid-channel expansion to 32 — it's a spectro-temporal conv with very few channels.

## Suggested fixes

Two options, in order of how I'd lean:

1. **Drop the kernel-size framing entirely.** Re-label the figure rows as `Channel-mixing 1-D` (or `Temporal channel-mixing`) and `Spectro-temporal 2-D`, and update the prose to say:

   > Two variants are tested: a 1-D causal conv that mixes mel bins through an 80→80 channel matrix at each time step (with a 3-frame temporal kernel), and a 2-D spectro-temporal conv with a 3 × 3 kernel that mixes both axes through a 1→32→1 channel bottleneck.

   The param contrast then reads naturally from the channel structure.

2. **Keep kernel-area framing but get it right.** Re-label as `1-D, k=3` / `2-D, k=3 × 3` and adjust the prose to remove the `1 × 1` claim. Less informative because both have kernel=3 along time — the kernel-size axis no longer separates them.

The chart's right-side `(+38.9K)` / `(+0.7K)` annotations are correct under both options and need no change.
