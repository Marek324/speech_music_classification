# tests/test_common.py

import pytest
from src.common import frame_label, frame_label_str, ms_to_samples, LABEL_MAP


# ── frame_label (uses dict-keyed annotation format) ────────────────────────

def _ann(lbl, start, end):
    """Build annotation in the format expected by frame_label."""
    return {lbl: {"start": start, "end": end}}


def test_frame_label_empty_anns():
    assert frame_label([], 0, 100) == LABEL_MAP["inactive"]


def test_frame_label_no_overlap():
    # annotation is outside the frame window
    ann = _ann("speech", 200, 300)
    assert frame_label([ann], 0, 100) == LABEL_MAP["inactive"]


def test_frame_label_majority_speech():
    # speech covers 80ms, music covers 20ms
    anns = [_ann("speech", 0, 80), _ann("music", 80, 100)]
    assert frame_label(anns, 0, 100) == LABEL_MAP["speech"]


def test_frame_label_majority_music():
    anns = [_ann("music", 0, 90), _ann("speech", 90, 100)]
    assert frame_label(anns, 0, 100) == LABEL_MAP["music"]


def test_frame_label_partial_overlap():
    # annotation starts before the frame and overlaps partially
    ann = _ann("speech", 50, 200)
    result = frame_label([ann], 100, 200)
    assert result == LABEL_MAP["speech"]


# ── frame_label_str (uses label/start/end annotation format) ──────────────

def _ann_str(lbl, start, end):
    return {"label": lbl, "start": start, "end": end}


def test_frame_label_str_returns_string():
    ann = _ann_str("speech", 0, 100)
    result = frame_label_str([ann], 0, 100)
    assert isinstance(result, str)


def test_frame_label_str_empty():
    assert frame_label_str([], 0, 100) == "inactive"


def test_frame_label_str_music():
    ann = _ann_str("music", 0, 100)
    assert frame_label_str([ann], 0, 100) == "music"


def test_frame_label_str_speech():
    ann = _ann_str("speech", 0, 100)
    assert frame_label_str([ann], 0, 100) == "speech"


def test_frame_label_str_unknown_label_raises():
    ann = _ann_str("unknown_cls", 0, 100)
    with pytest.raises(ValueError):
        frame_label_str([ann], 0, 100)


def test_frame_label_str_majority():
    anns = [_ann_str("speech", 0, 70), _ann_str("music", 70, 100)]
    assert frame_label_str(anns, 0, 100) == "speech"


# ── ms_to_samples ──────────────────────────────────────────────────────────

def test_ms_to_samples_basic():
    assert ms_to_samples(1000, 22050) == 22050


def test_ms_to_samples_zero():
    assert ms_to_samples(0, 22050) == 0


def test_ms_to_samples_fractional_rounds_down():
    # 1ms at 16000 Hz = 16 samples
    assert ms_to_samples(1, 16000) == 16
