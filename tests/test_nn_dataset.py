# tests/test_nn_dataset.py

import numpy as np
import pytest
import torch
from src.nn.dataset import _class_to_target, _timestamps_to_frame_labels, iter_nn_rows


# ── _class_to_target ──────────────────────────────────────────────────────

def test_class_to_target_speech():
    t = _class_to_target("speech")
    assert t.tolist() == [1.0, 0.0, 0.0]


def test_class_to_target_music():
    t = _class_to_target("music")
    assert t.tolist() == [0.0, 1.0, 0.0]


def test_class_to_target_inactive():
    t = _class_to_target("inactive")
    assert t.tolist() == [0.0, 0.0, 1.0]


def test_class_to_target_noise_maps_to_inactive():
    t = _class_to_target("noise")
    assert t.tolist() == [0.0, 0.0, 1.0]


def test_class_to_target_unknown_maps_to_inactive():
    t = _class_to_target("something_unknown")
    assert t.tolist() == [0.0, 0.0, 1.0]


# ── _timestamps_to_frame_labels ──────────────────────────────────────────

SR = 22050
HOP = 512


def test_timestamps_to_frame_labels_full_speech_coverage():
    n_frames = 10
    duration_ms = int(n_frames * HOP / SR * 1000) + 100  # longer than clip
    labels = [{"label": "speech", "start": 0, "end": duration_ms}]
    fl = _timestamps_to_frame_labels(labels, n_frames, SR, HOP)
    assert fl.shape == (n_frames,)
    assert (fl == 0).all()  # 0 = speech


def test_timestamps_to_frame_labels_no_coverage():
    n_frames = 10
    fl = _timestamps_to_frame_labels([], n_frames, SR, HOP)
    assert (fl == -1).all()  # -1 = inactive


def test_timestamps_to_frame_labels_music():
    n_frames = 10
    duration_ms = int(n_frames * HOP / SR * 1000) + 100
    labels = [{"label": "music", "start": 0, "end": duration_ms}]
    fl = _timestamps_to_frame_labels(labels, n_frames, SR, HOP)
    assert (fl == 1).all()  # 1 = music


def test_timestamps_to_frame_labels_partial():
    n_frames = 20
    # Cover frames 0-4 only: end_ms just past frame 5 boundary
    end_frame = 5
    end_ms = int(end_frame * HOP / SR * 1000) + 1  # slightly past frame 5 start → covers [0,5)
    labels = [{"label": "speech", "start": 0, "end": end_ms}]
    fl = _timestamps_to_frame_labels(labels, n_frames, SR, HOP)
    # frames before end_frame should be speech (0)
    assert (fl[:end_frame] == 0).all()
    # frames beyond should remain inactive (-1)
    assert (fl[end_frame:] == -1).all()


def test_timestamps_to_frame_labels_unknown_label_skipped():
    n_frames = 5
    labels = [{"label": "unknown", "start": 0, "end": 9999}]
    fl = _timestamps_to_frame_labels(labels, n_frames, SR, HOP)
    assert (fl == -1).all()


# ── iter_nn_rows ─────────────────────────────────────────────────────────

N_FFT = 1024


def _make_row(n_samples, cls="speech", labels=None):
    return {
        "audio": {"array": np.zeros(n_samples, dtype=np.float32)},
        "class": cls,
        "labels": labels or [],
    }


def test_iter_nn_rows_yields_wav_and_targets():
    rows = [_make_row(SR * 3)]
    results = list(iter_nn_rows(rows, None, "test", SR, HOP, N_FFT))
    assert len(results) == 1
    wav, targets = results[0]
    assert wav.shape[0] == 1  # mono


def test_iter_nn_rows_target_shape():
    rows = [_make_row(SR * 3)]
    wav, targets = next(iter(iter_nn_rows(rows, None, "test", SR, HOP, N_FFT)))
    n_frames = (SR * 3 - N_FFT) // HOP + 1
    assert targets.shape == (3, n_frames)


def test_iter_nn_rows_short_clip_skipped():
    # shorter than 1 frame
    rows = [_make_row(N_FFT - 1)]
    results = list(iter_nn_rows(rows, None, "test", SR, HOP, N_FFT))
    assert results == []


def test_iter_nn_rows_mono_from_2d():
    # 2-channel audio stored as (N, 2) — time-first, channels-second
    n = SR * 2
    audio_2ch = np.zeros((n, 2), dtype=np.float32)
    rows = [{"audio": {"array": audio_2ch}, "class": "music", "labels": []}]
    results = list(iter_nn_rows(rows, None, "test", SR, HOP, N_FFT))
    assert len(results) == 1
    wav, _ = results[0]
    assert wav.shape[0] == 1


def test_iter_nn_rows_max_rows():
    rows = [_make_row(SR * 3)] * 5
    results = list(iter_nn_rows(rows, 2, "test", SR, HOP, N_FFT))
    assert len(results) == 2


def test_iter_nn_rows_label_format_list_of_dicts():
    n = SR * 3
    duration_ms = int(3 * 1000)
    labels = [{"label": "speech", "start": 0, "end": duration_ms}]
    rows = [_make_row(n, labels=labels)]
    wav, targets = next(iter(iter_nn_rows(rows, None, "test", SR, HOP, N_FFT)))
    # speech frames should have targets[0] = 1
    assert targets[0].max().item() == 1.0


def test_iter_nn_rows_label_format_dict_of_lists():
    n = SR * 3
    duration_ms = int(3 * 1000)
    labels = {"label": ["speech"], "start": [0], "end": [duration_ms]}
    rows = [{"audio": {"array": np.zeros(n, dtype=np.float32)}, "class": "speech", "labels": labels}]
    wav, targets = next(iter(iter_nn_rows(rows, None, "test", SR, HOP, N_FFT)))
    assert targets.shape[0] == 3


def test_iter_nn_rows_yields_subclass():
    rows = [{"audio": {"array": np.zeros(SR * 2, dtype=np.float32)},
             "class": "speech", "subclass": "speech_clean", "labels": []}]
    results = list(iter_nn_rows(rows, None, "test", SR, HOP, N_FFT, yield_subclass=True))
    assert len(results) == 1
    wav, targets, cls, sub = results[0]
    assert sub == "speech_clean"
    assert cls == "speech"
