# feat_extractor.py
# Marek Hric

import warnings
from collections import deque
from copy import deepcopy
from typing import Literal, Optional

import numpy as np
from librosa import feature as libfeat
from librosa.filters import mel as mel_filter_bank
from librosa.util import fix_length
from librosa import lpc
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, lfilter, find_peaks, hilbert
from scipy.stats import skew
from scipy.fft import fft

from .. import config

FeatureSet = Literal["decision_tree", "gmm_svm"]

N_FFT = "n_fft"
BUFFERS = "buffers"
FEATURES = "features"


def _safe(arr: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf with 0."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _stack_subframes(segment: np.ndarray, positions: np.ndarray, fl: int) -> np.ndarray:
    """Build a (len(positions), fl) array where row i is segment[positions[i]:positions[i]+fl]."""
    if positions.size == 0:
        return np.zeros((0, fl), dtype=segment.dtype)
    idx = positions[:, None] + np.arange(fl)[None, :]
    return segment[idx]


class FeatExtractor:
    def __init__(self, sec_buffer_scale: int = 30) -> None:
        self.cfg = config.get_config()
        buf = self.cfg[BUFFERS]
        self.sr = buf.get("sample_rate", self.cfg["sample_rate"])
        self.n_fft = self.cfg[N_FFT]
        self.fl = buf["frame_length_ms"] * self.sr // 1000
        self.fh = buf["hop_length_ms"] * self.sr // 1000
        self.lt_len = int(buf["lt_len_ms"] * self.sr // 1000)

        self.signal_buffer: deque[float] = deque(maxlen=self.lt_len)
        self.feat_buf_size = int(buf["lt_len_ms"] / buf["hop_length_ms"]) - 1
        self.feat_buffer: deque[np.ndarray] = deque(maxlen=self.feat_buf_size)
        self.sec_buf_size = self.feat_buf_size * sec_buffer_scale
        self.sec_feat_buffer: deque[np.ndarray] = deque(maxlen=self.sec_buf_size)

        self.last_fft: Optional[np.ndarray] = None
        self.last_mfcc: Optional[np.ndarray] = None

        # Precomputed constants for vectorized subframe features.
        self._mel_fb_logmel = mel_filter_bank(
            sr=self.sr, n_fft=self.n_fft, n_mels=22, fmin=0, fmax=4000
        ).astype(np.float64)  # (22, n_fft//2+1), speech-specific log-mel feature
        self._mel_fb_mod = mel_filter_bank(
            sr=self.sr, n_fft=self.n_fft, n_mels=18, fmin=0, fmax=self.sr // 2
        ).astype(np.float64)  # (18, n_fft//2+1), modulation spectrum feature
        self._hann_nfft = np.hanning(self.n_fft).astype(np.float64)
        self._step_1ms = max(1, int(self.sr * 0.001))
        self._psr_step_ms = 10
        self._psr_step = max(1, int(self.sr * self._psr_step_ms / 1000))
        self._mod_hop = max(1, self.sr // 80)         # 100 samples at 8kHz → 80Hz
        self._mod_win_len = 20                         # 250ms @ 80Hz

        # Streaming state (used only by the _extract_gmm_svm per-hop path).
        self._stream_signal = np.zeros(self.lt_len, dtype=np.float64)
        self._stream_filled = 0
        self._stream_total = 0
        self._stream_next_1ms_pos = 0
        self._stream_next_psr_pos = 0
        self._stream_next_mod_pos = 0
        ring_1ms = (self.lt_len - self.fl) // self._step_1ms + 1
        ring_psr = (self.lt_len - self.fl) // self._psr_step + 1
        ring_mod = (self.lt_len - self.n_fft) // self._mod_hop + 1
        self._stream_naps_ring: deque[float] = deque(maxlen=ring_1ms)
        self._stream_logmel_ring: deque[float] = deque(maxlen=ring_1ms)
        self._stream_psr_ring: deque[float] = deque(maxlen=ring_psr)
        self._stream_mod_ring: deque[np.ndarray] = deque(maxlen=ring_mod)

    def reset(self) -> None:
        """Reset all stateful buffers between clips."""
        self.signal_buffer.clear()
        self.feat_buffer.clear()
        self.sec_feat_buffer.clear()
        self.last_fft = None
        self.last_mfcc = None

        self._stream_signal[:] = 0.0
        self._stream_total = 0
        self._stream_next_1ms_pos = 0
        self._stream_next_psr_pos = 0
        self._stream_next_mod_pos = 0
        self._stream_naps_ring.clear()
        self._stream_logmel_ring.clear()
        self._stream_psr_ring.clear()
        self._stream_mod_ring.clear()

    def _get_feature_set(self) -> FeatureSet:
        features = self.cfg.get(FEATURES, {})
        name = features.get("set") or self.cfg["model"]["name"]
        return "gmm_svm" if name in ("gmm", "svm") else "decision_tree"

    def _frame_for_fft(self, frame: np.ndarray) -> np.ndarray:
        return fix_length(frame, size=self.n_fft)

    def _get_sig_buffer(self) -> np.ndarray:
        buffer = self.signal_buffer
        out = np.zeros(self.lt_len, dtype=float)
        out[-len(buffer) :] = np.fromiter(buffer, dtype=float)
        return out

    def _get_feat_buffer(self, secondary: bool = False) -> np.ndarray:
        buf = self.sec_feat_buffer if secondary else self.feat_buffer
        if not buf:
            return np.empty((0, 0), dtype=float)
        return np.asarray(buf, dtype=float)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        self.signal_buffer.extend(frame[-self.fh :])
        feature_set = self._get_feature_set()
        if feature_set == "decision_tree":
            return self._extract_decision_tree(frame)
        return self._extract_gmm_svm(frame)

    def extract_segment(self, segment: np.ndarray) -> np.ndarray:
        """Extract one 9-dim feature vector from a 1s audio segment (paper §3.2).

        Vectorized and speed-adapted version of Khonglah & Prasanna 2016:
          - NAPS at 1ms shift, batched via stacked lfilter + FFT autocorr
          - PSR at 10ms shift (LPC is recursive — coarsened; parity corr > 0.99)
          - log-mel at 1ms shift via batched manual mel filterbank on zero-padded subframes
          - Modulation spectrum unchanged (segment-level, already batched)
        """
        n = len(segment)
        fl = self.fl

        # --- Step 1: per-frame existing features (30ms / 15ms hop) ---
        var_zcr, var_centroid, var_flux, var_rolloff, lster = self._existing_feature_stats(
            segment
        )

        # --- Step 2: batched speech-specific features ---
        # Build subframe index tables once, share across NAPS / PSR / log-mel.
        last_valid = n - fl
        positions_1ms = np.arange(0, last_valid + 1, self._step_1ms, dtype=np.int64)
        subframes_1ms = _stack_subframes(segment, positions_1ms, fl)  # (N1, fl)

        naps_vals = self._batched_naps_of_zffs(subframes_1ms)
        logmel_vals = self._batched_log_mel_energy(subframes_1ms)

        positions_psr = np.arange(0, last_valid + 1, self._psr_step, dtype=np.int64)
        psr_vals = np.empty(len(positions_psr), dtype=np.float64)
        for i, pos in enumerate(positions_psr):
            psr_vals[i] = self._psr_he_lp_residual(segment[pos : pos + fl])

        mean_naps = float(np.mean(naps_vals)) if naps_vals.size else 0.0
        mean_psr = float(np.mean(psr_vals)) if psr_vals.size else 0.0
        var_mel = float(np.var(logmel_vals)) if logmel_vals.size else 0.0

        # --- Step 3: segment-level modulation spectrum ---
        mod_energy = self._modulation_spectrum_energy_segment(segment)

        return _safe(np.array([
            var_zcr, var_centroid, var_flux, var_rolloff, lster,
            mean_naps, mean_psr, var_mel, mod_energy,
        ]))

    def _existing_feature_stats(self, segment: np.ndarray) -> tuple:
        """Step 1 of extract_segment: 30ms-frame features over 15ms hop.

        Returns (var_zcr, var_centroid, var_flux, var_rolloff, lster).
        Not a bottleneck — only ~66 frames per 1s segment — so left as a loop.
        """
        n = len(segment)
        frames_feats = []
        last_fft = None
        srp_thresh = self.cfg[FEATURES]["spectral_rolloff_point"]["threshold"]
        pos = 0
        while pos + self.fl <= n:
            frame = segment[pos : pos + self.fl]
            padded = fix_length(frame, size=self.n_fft)

            ste = float(10 * np.log10(np.mean(frame**2) + 1e-10))
            zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / 2)

            sc = libfeat.spectral_centroid(
                y=padded, sr=self.sr, n_fft=self.n_fft
            )[0, 0]
            sc = 0.0 if np.isnan(sc) or np.isinf(sc) else float(sc)

            cur_fft = np.fft.fft(padded, n=self.n_fft)
            flux = 0.0 if last_fft is None else float(
                np.sum(np.abs(cur_fft - last_fft) ** 2)
            )
            last_fft = cur_fft

            rolloff = float(
                libfeat.spectral_rolloff(
                    y=padded, sr=self.sr, roll_percent=srp_thresh, n_fft=self.n_fft
                )[0, 0]
            )
            frames_feats.append([ste, zcr, sc, flux, rolloff])
            pos += self.fh

        if not frames_feats:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        arr = np.array(frames_feats)
        var_zcr = float(np.var(arr[:, 1]))
        var_centroid = float(np.var(arr[:, 2]))
        var_flux = float(np.var(arr[:, 3]))
        var_rolloff = float(np.var(arr[:, 4]))
        energies = arr[:, 0]
        thr = float(np.mean(energies) / 3)
        lster = float(np.sum(energies < thr) / len(energies))
        return var_zcr, var_centroid, var_flux, var_rolloff, lster

    def _batched_naps_of_zffs(self, subframes: np.ndarray) -> np.ndarray:
        """Vectorized NAPS. subframes: (N, fl) float. Returns (N,).

        Semantically equivalent to looping _naps_of_zffs over each row, modulo
        scipy FFT vs direct-correlate rounding (< 1e-8 abs diff).
        """
        if subframes.size == 0:
            return np.zeros(0, dtype=np.float64)
        N, L = subframes.shape
        subframes = np.ascontiguousarray(subframes, dtype=np.float64)

        # diff with prepend (per row): out[0] = 0, out[k>0] = sub[k] - sub[k-1]
        diffs = np.diff(subframes, axis=-1, prepend=subframes[:, :1])
        # IIR cascade (ZFFS)
        y = lfilter([1.0], [1.0, -4.0, 6.0, -4.0, 1.0], diffs, axis=-1)
        # Two 10ms trend-removal passes
        N_uf = int(self.sr * 0.01)
        y_1 = y - uniform_filter1d(y, size=N_uf, axis=-1, mode="nearest")
        zffs = y_1 - uniform_filter1d(y_1, size=N_uf, axis=-1, mode="nearest")
        # Input is already length fl, so [-fl:] is a no-op. Keep it explicit for clarity.
        zffs = zffs[:, -L:]

        # Batched linear autocorrelation via FFT, matching the
        # correlate(a,a,'full')[L//2:] slice used in the scalar version.
        pad = 2 * L - 1
        n_fft = 1 << (pad - 1).bit_length()
        Z = np.fft.rfft(zffs, n=n_fft, axis=-1)
        auto = np.fft.irfft(Z * np.conj(Z), n=n_fft, axis=-1)
        neg = auto[:, n_fft - (L - 1) : n_fft]
        pos = auto[:, :L]
        full = np.concatenate([neg, pos], axis=-1)  # length 2L-1
        r_sliced = full[:, L // 2 :]                # matches scalar slicing

        r0 = r_sliced[:, :1]
        r0_safe = np.where(r0 != 0, r0, 1.0)
        r_norm = np.where(r0 != 0, r_sliced / r0_safe, 0.0)

        min_dist = int(self.sr * 0.002)
        out = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if r0[i, 0] == 0:
                continue
            peaks, _ = find_peaks(r_norm[i], distance=min_dist)
            if peaks.size:
                out[i] = float(r_norm[i, peaks[0]])
        return out

    def _batched_log_mel_energy(self, subframes: np.ndarray) -> np.ndarray:
        """Vectorized log-mel energy. subframes: (N, fl). Returns (N,).

        Deviation from the scalar version: one windowed FFT per subframe
        (single analysis frame of n_fft samples, 240 real + zero-pad to 512),
        instead of librosa's internal 5 center-padded frames. Simpler and
        ~50x faster; retraining absorbs the small distribution shift.
        """
        if subframes.size == 0:
            return np.zeros(0, dtype=np.float64)
        N, fl = subframes.shape
        padded = np.zeros((N, self.n_fft), dtype=np.float64)
        padded[:, :fl] = subframes
        padded *= self._hann_nfft
        X = np.fft.rfft(padded, n=self.n_fft, axis=-1)
        P = (X.real ** 2 + X.imag ** 2).astype(np.float64)  # (N, n_fft//2+1)
        M = P @ self._mel_fb_logmel.T  # (N, 22)
        log_mel = np.log(M[:, :18] + 1e-10)
        return np.sum(log_mel, axis=-1)

    def _modulation_spectrum_energy_segment(self, segment: np.ndarray) -> float:
        """4 Hz modulation spectrum energy across 18 mel bands (paper §2.3.1).

        Center=False manual mel cols + shared energy aggregator so the streaming
        path produces the same output as the batch segment path.
        """
        cols = self._compute_mod_mel_cols_batch(segment)
        return self._compute_mod_energy_from_cols(cols)

    def _compute_mod_mel_cols_batch(self, segment: np.ndarray) -> np.ndarray:
        """Return (18, T) power-mel cols on a center=False grid, hop=_mod_hop."""
        n = len(segment)
        hop = self._mod_hop
        last = n - self.n_fft
        if last < 0:
            return np.zeros((18, 0), dtype=np.float64)
        positions = np.arange(0, last + 1, hop, dtype=np.int64)
        idx = positions[:, None] + np.arange(self.n_fft, dtype=np.int64)[None, :]
        frames = segment[idx].astype(np.float64, copy=False) * self._hann_nfft
        X = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        P = X.real ** 2 + X.imag ** 2
        M = P @ self._mel_fb_mod.T  # (T, 18)
        return M.T

    def _compute_mod_energy_from_cols(self, S: np.ndarray) -> float:
        """Run the paper's 250ms Hamming DFT across mel cols; return 4 Hz energy."""
        win_len = self._mod_win_len
        T = S.shape[1]
        if T < win_len:
            return 0.0
        band_means = S.mean(axis=1, keepdims=True)
        S_norm = S / (band_means + 1e-10)
        n_windows = T - win_len + 1
        idx = np.arange(n_windows)[:, None] + np.arange(win_len)[None, :]
        windowed = S_norm[:, idx] * np.hamming(win_len)[None, None, :]
        fft_vals = np.fft.fft(windowed, axis=-1)
        mod_energy = float(np.sum(np.abs(fft_vals[:, :, 1]) ** 2))
        return mod_energy / n_windows

    def _extract_decision_tree(self, frame: np.ndarray) -> np.ndarray:
        feats = [
            self._short_time_energy(frame),
            self._zero_crossing_rate(frame),
            self._autocorrelation_coefficient(frame),
        ]
        mfccs = _safe(self._mfcc(frame).flatten())
        feats.extend(mfccs)
        feats.append(self._mfcc_diff_norm(mfccs))
        self.last_mfcc = deepcopy(mfccs)

        feats.extend([
            self._band_energy_ratio(frame),
            self._spectral_rolloff_point(frame),
            self._spectrum_centroid(frame),
            self._spectrum_spread(frame),
            self._spectral_flux(frame),
        ])

        feats = np.array(feats)
        self.feat_buffer.append(feats)

        buf = self._get_feat_buffer()
        energies, zcrs = buf[:, 0], buf[:, 1]
        diffs = np.abs(np.diff(buf, axis=0))

        feats = np.hstack([
            feats,
            self._stats_mean(buf),
            self._stats_std(buf),
            self._stats_mean(diffs),
            self._stats_std(diffs),
            self._stats_skew(zcrs),
            self._stats_skew(diffs[:, 1]),
            self._low_short_time_energy_ratio(energies),
        ])
        return _safe(feats)

    def _extract_gmm_svm(self, frame: np.ndarray) -> np.ndarray:
        """Streaming 9-D feature on every 15ms hop, aggregating over last 1s.

        Semantically tracks extract_segment: the same per-subframe quantities
        populate ring buffers sized to the 1s window, and aggregate statistics
        are recomputed on every call. During the first ~1s of a clip the
        rings are still filling, so early outputs are partial-stats best-effort.
        """
        new_samples = np.asarray(frame[-self.fh :], dtype=np.float64)
        self._advance_stream(new_samples)

        # (1) base per-30ms features for the newest fully-filled frame.
        if self._stream_total >= self.fl:
            base_local = self.lt_len - self.fl
            base_frame = self._stream_signal[base_local : base_local + self.fl]
            self.feat_buffer.append(np.array([
                self._short_time_energy(base_frame),
                self._zero_crossing_rate(base_frame),
                self._spectrum_centroid(base_frame),
                self._spectral_flux(base_frame),
                self._spectral_rolloff_point(base_frame),
            ], dtype=np.float64))

        # (2) NAPS + log-mel at 1ms shift — batched over any new positions.
        new_1ms = self._collect_new_positions("_stream_next_1ms_pos", self._step_1ms, self.fl)
        if new_1ms.size:
            subs = self._subframes_at(new_1ms, self.fl)
            naps_new = self._batched_naps_of_zffs(subs)
            logmel_new = self._batched_log_mel_energy(subs)
            self._stream_naps_ring.extend(naps_new.tolist())
            self._stream_logmel_ring.extend(logmel_new.tolist())

        # (3) PSR at 10ms shift — LPC is recursive, loop is fine.
        new_psr = self._collect_new_positions("_stream_next_psr_pos", self._psr_step, self.fl)
        if new_psr.size:
            shift = self.lt_len - self._stream_total
            for p in new_psr:
                local = int(p) + shift
                self._stream_psr_ring.append(
                    self._psr_he_lp_residual(self._stream_signal[local : local + self.fl])
                )

        # (4) modulation mel cols at hop=_mod_hop, window=n_fft.
        new_mod = self._collect_new_positions("_stream_next_mod_pos", self._mod_hop, self.n_fft)
        if new_mod.size:
            subs = self._subframes_at(new_mod, self.n_fft)
            windowed = subs * self._hann_nfft
            X = np.fft.rfft(windowed, n=self.n_fft, axis=-1)
            P = X.real ** 2 + X.imag ** 2
            mel_cols = P @ self._mel_fb_mod.T  # (T_new, 18)
            for col in mel_cols:
                self._stream_mod_ring.append(col)

        # (5) aggregate 9-D output.
        existing = self._get_feat_buffer()
        lster = (
            self._low_short_time_energy_ratio(existing[:, 0]) if existing.size else 0.0
        )
        var_zcr = float(np.var(existing[:, 1])) if existing.size else 0.0
        var_centroid = float(np.var(existing[:, 2])) if existing.size else 0.0
        var_flux = float(np.var(existing[:, 3])) if existing.size else 0.0
        var_rolloff = float(np.var(existing[:, 4])) if existing.size else 0.0

        mean_naps = float(np.mean(self._stream_naps_ring)) if self._stream_naps_ring else 0.0
        mean_psr = float(np.mean(self._stream_psr_ring)) if self._stream_psr_ring else 0.0
        var_mel = float(np.var(self._stream_logmel_ring)) if self._stream_logmel_ring else 0.0

        if self._stream_mod_ring:
            mod_cols = np.stack(list(self._stream_mod_ring), axis=1)  # (18, T)
            mod_energy = self._compute_mod_energy_from_cols(mod_cols)
        else:
            mod_energy = 0.0

        return _safe(np.array([
            var_zcr, var_centroid, var_flux, var_rolloff, lster,
            mean_naps, mean_psr, var_mel, mod_energy,
        ]))

    def _advance_stream(self, new_samples: np.ndarray) -> None:
        """Append new samples into the rolling 1s stream buffer."""
        k = new_samples.size
        if k == 0:
            return
        if k >= self.lt_len:
            self._stream_signal[:] = new_samples[-self.lt_len :]
        else:
            self._stream_signal[:-k] = self._stream_signal[k:]
            self._stream_signal[-k:] = new_samples
        self._stream_total += k

    def _collect_new_positions(self, next_attr: str, step: int, win_len: int) -> np.ndarray:
        """Absolute positions whose window [p:p+win_len] is now fully available."""
        next_pos = getattr(self, next_attr)
        last_valid = self._stream_total - win_len
        if last_valid < next_pos:
            return np.zeros(0, dtype=np.int64)
        positions = np.arange(next_pos, last_valid + 1, step, dtype=np.int64)
        if positions.size:
            setattr(self, next_attr, int(positions[-1] + step))
        return positions

    def _subframes_at(self, positions: np.ndarray, win_len: int) -> np.ndarray:
        """Stack _stream_signal windows at the given absolute positions.

        Data in _stream_signal is right-aligned so a single offset maps
        absolute → local for both warmup and post-warmup states.
        """
        local = positions + (self.lt_len - self._stream_total)
        idx = local[:, None] + np.arange(win_len, dtype=np.int64)[None, :]
        return self._stream_signal[idx]

    # --- Low-level feature extractors ---

    def _short_time_energy(self, frame: np.ndarray) -> float:
        return float(10 * np.log10(np.mean(frame**2) + 1e-10))

    def _zero_crossing_rate(self, frame: np.ndarray) -> float:
        return float(np.sum(np.abs(np.diff(np.sign(frame)))) / 2)

    def _band_energy_ratio(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        berconf = self.cfg[FEATURES]["band_energy_ratio"]
        low, high = berconf["low"], berconf["high"]

        def band_energy(dft: np.ndarray, lb: float, ub: float) -> float:
            K = dft.shape[0]
            b1 = int(np.floor(K * lb / self.sr))
            b2 = int(np.floor(K * ub / self.sr))
            return float(np.sum(np.abs(dft[b1:b2]) ** 2))

        dft = np.fft.fft(frame, n=self.n_fft)
        nyquist = self.sr / 2
        E1 = band_energy(dft, 0.0, low)
        E2 = band_energy(dft, high, nyquist)
        ratio = np.clip((E1 + 1e-10) / (E2 + 1e-10), 1e-10, 1e10)
        result = 10 * np.log10(ratio)
        return 0.0 if np.isnan(result) or np.isinf(result) else float(result)

    def _autocorrelation_coefficient(self, frame: np.ndarray) -> float:
        acconf = self.cfg[FEATURES]["autocorrelation_coefficient"]
        ac = correlate(frame, frame, mode="full")[frame.shape[0] // 2 :]
        min_lag = acconf["min_lag_ms"] * self.sr // 1000
        max_lag = acconf["max_lag_ms"] * self.sr // 1000
        return float(np.max(ac[min_lag:max_lag]))

    def _spectral_rolloff_point(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        srp = self.cfg[FEATURES]["spectral_rolloff_point"]
        return float(
            libfeat.spectral_rolloff(
                y=frame, sr=self.sr, roll_percent=srp["threshold"], n_fft=self.n_fft
            )[0, 0]
        )

    def _spectrum_centroid(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        val = libfeat.spectral_centroid(y=frame, sr=self.sr, n_fft=self.n_fft)[0, 0]
        return 0.0 if np.isnan(val) or np.isinf(val) else float(val)

    def _spectrum_spread(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        val = libfeat.spectral_bandwidth(y=frame, sr=self.sr, n_fft=self.n_fft)[0, 0]
        return 0.0 if np.isnan(val) or np.isinf(val) else float(val)

    def _spectral_flux(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        current_fft = np.fft.fft(frame, n=self.n_fft)
        if self.last_fft is None:
            self.last_fft = current_fft
            return 0.0
        flux = float(np.sum(np.abs(current_fft - self.last_fft) ** 2))
        self.last_fft = current_fft
        return flux

    def _mfcc(self, frame: np.ndarray) -> np.ndarray:
        frame = self._frame_for_fft(frame)
        return libfeat.mfcc(
            y=frame, sr=self.sr, n_mfcc=10, hop_length=self.fh, n_fft=self.n_fft
        )

    def _mfcc_diff_norm(self, mfccs: np.ndarray) -> float:
        if self.last_mfcc is None:
            return 0.0
        return float(np.sqrt(np.sum(np.abs(mfccs - self.last_mfcc) ** 2)))

    def _low_short_time_energy_ratio(self, energies: np.ndarray) -> float:
        if len(energies) == 0:
            return 0.0
        thr = float(np.mean(energies) / 3)
        return float(np.sum(energies < thr) / len(energies))

    def _naps_of_zffs(self, frame: np.ndarray) -> float:
        diff = np.diff(frame, prepend=frame[0])
        y = lfilter([1], [1, -4, 6, -4, 1], diff)
        N = int(self.sr * 0.01)
        y_1 = y - uniform_filter1d(y, size=N, mode="nearest")
        zffs = y_1 - uniform_filter1d(y_1, size=N, mode="nearest")
        zffs = zffs[-self.fl :]

        r = correlate(zffs, zffs, mode="full")[zffs.shape[0] // 2 :]
        if r[0] == 0:
            return 0.0
        r = r / r[0]
        peaks, _ = find_peaks(r, distance=int(self.sr * 0.002))
        return 0.0 if len(peaks) == 0 else float(r[peaks[0]])

    def _psr_he_lp_residual(self, frame: np.ndarray) -> float:
        a = lpc(frame, order=10)
        residual = lfilter([1], a, frame)
        he = np.abs(hilbert(residual))
        min_dist = int(self.sr * 0.005)
        peaks, _ = find_peaks(he, distance=min_dist)
        if len(peaks) == 0:
            return 0.0

        peak_idx = int(peaks[np.argmax(he[peaks])])
        peak_val = he[peak_idx]
        half = int(self.sr * 0.01) // 2
        left = he[max(0, peak_idx - half) : max(0, peak_idx - 4)]
        right = he[min(len(he), peak_idx + 4) : min(len(he), peak_idx + half)]
        sidelobes = np.concatenate((left, right))
        if sidelobes.size == 0:
            return 0.0
        return float(peak_val / (np.var(sidelobes) + 1e-4))

    def _log_mel_spectrum_energy(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        S = libfeat.melspectrogram(
            y=frame, sr=self.sr, n_fft=self.n_fft,
            n_mels=22, fmin=0, fmax=4000, power=2.0,
        )
        return float(np.sum(np.log(S + 1e-10)[:18, :]))

    def _stats_mean(self, feats: np.ndarray) -> np.ndarray:
        return self._stats_agg(feats, np.mean)

    def _stats_std(self, feats: np.ndarray) -> np.ndarray:
        return self._stats_agg(feats, np.std)

    def _stats_var(self, feats: np.ndarray) -> np.ndarray:
        return self._stats_agg(feats, np.var)

    def _stats_skew(self, feats: np.ndarray) -> np.ndarray:
        if feats.size == 0 or feats.shape[0] < 3:
            return self._empty_like(feats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nan_to_num(skew(feats, axis=0), nan=0.0)

    def _stats_agg(self, feats: np.ndarray, agg) -> np.ndarray:
        if feats.size == 0 or feats.shape[0] < 1:
            return self._empty_like(feats)
        return agg(feats, axis=0)

    def _empty_like(self, feats: np.ndarray) -> np.ndarray:
        dim = 1 if feats.ndim == 1 else feats.shape[1]
        return np.zeros(dim)
