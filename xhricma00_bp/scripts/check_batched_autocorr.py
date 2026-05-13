# scripts/check_batched_autocorr.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Verify batched FFT autocorrelation matches scipy.signal.correlate 'full' slice."""
import numpy as np
from scipy.signal import correlate

rng = np.random.default_rng(0)
N, L = 5, 240
X = rng.standard_normal((N, L)).astype(np.float64)


def ref(x):
    return correlate(x, x, mode="full")[x.shape[0] // 2 :]


def batched_correlate_full_slice(X):
    """Returns (N, 3L/2 - 1) matching ref() for each row."""
    L = X.shape[-1]
    pad = 2 * L - 1
    n_fft = 1 << (pad - 1).bit_length()
    Z = np.fft.rfft(X, n=n_fft, axis=-1)
    auto = np.fft.irfft(Z * np.conj(Z), n=n_fft, axis=-1)
    neg = auto[:, n_fft - (L - 1) : n_fft]
    pos = auto[:, :L]
    full = np.concatenate([neg, pos], axis=-1)
    return full[:, L // 2 :]


out_ref = np.stack([ref(X[i]) for i in range(N)])
out_batched = batched_correlate_full_slice(X)
print("shapes:", out_ref.shape, out_batched.shape)
print("max abs diff:", np.max(np.abs(out_ref - out_batched)))
print("ok:", np.allclose(out_ref, out_batched, atol=1e-8))
