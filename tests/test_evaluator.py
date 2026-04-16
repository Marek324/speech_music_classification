# tests/test_evaluator.py

import numpy as np
import pytest
from src.evaluator import (
    run_evaluation,
    format_report,
    _compute_subclass_metrics,
    EvalResults,
)


def _make_arrays(n=100):
    """Balanced arrays with labels -1, 1, 2."""
    rng = np.random.default_rng(0)
    y_true = np.tile([-1, 1, 2], n // 3 + 1)[:n]
    subclasses = np.array(["speech_clean"] * (n // 3) + ["music_pop"] * (n // 3) + ["noise"] * (n - 2 * (n // 3)))
    return y_true, subclasses


def test_run_evaluation_perfect():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 1000.0, "test_perfect", save_to_file=False)
    assert abs(res.f1 - 1.0) < 1e-6


def test_run_evaluation_returns_eval_results():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 1000.0, "test_type", save_to_file=False)
    assert isinstance(res, EvalResults)


def test_run_evaluation_f1_below_one_when_wrong():
    y_true, subs = _make_arrays(99)
    y_pred = np.roll(y_true, 1)  # shift labels so nothing matches
    res = run_evaluation(y_true, y_pred, subs, 1000.0, "test_wrong", save_to_file=False)
    assert res.f1 < 1.0


def test_run_evaluation_per_class_has_all_labels():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 0.0, "test_labels", save_to_file=False)
    assert set(res.per_class.keys()) == {-1, 1, 2}


def test_run_evaluation_saves_file():
    import src.evaluator as ev_mod
    y_true, subs = _make_arrays(30)
    run_evaluation(y_true, y_true.copy(), subs, 0.0, "_pytest_tmp_save", save_to_file=True)
    expected = ev_mod.Path(__file__).resolve().parent.parent / "results" / "_pytest_tmp_save.eval"
    try:
        assert expected.exists()
    finally:
        expected.unlink(missing_ok=True)


def test_run_evaluation_no_file(tmp_path):
    y_true, subs = _make_arrays(30)
    run_evaluation(y_true, y_true.copy(), subs, 0.0, "_pytest_no_file", save_to_file=False)
    from pathlib import Path
    import src.evaluator as ev_mod
    p = ev_mod.Path(__file__).resolve().parent.parent / "results" / "_pytest_no_file.eval"
    assert not p.exists()


def test_run_evaluation_length_mismatch_raises():
    y_true = np.array([-1, 1, 2, -1])
    y_pred = np.array([-1, 1])
    subs = np.array(["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        run_evaluation(y_true, y_pred, subs, 0.0, "mismatch", save_to_file=False)


def test_f1_property_equals_mean_of_per_class():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 0.0, "f1_prop", save_to_file=False)
    manual = np.mean([res.per_class[l].f1 for l in res.labels])
    assert abs(res.f1 - manual) < 1e-9


def test_format_report_contains_f1():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 0.0, "fmt", save_to_file=False)
    report = format_report(res)
    assert "F1" in report or "f1" in report.lower()


def test_subclass_metrics():
    # speech_clean: 2 speech + 1 inactive frame; music_pop: 2 music + 1 inactive frame
    y_true = np.array([-1, -1,  2,  1,  1,  2])
    y_pred = np.array([-1, -1,  2,  1,  1,  1])  # music_pop: inactive frame mispredicted as music
    subs  = np.array(["speech_clean"] * 3 + ["music_pop"] * 3)
    by_sub = _compute_subclass_metrics(y_true, y_pred, subs)
    # speech_clean: perfect predictions → F1=1.0, P=1.0, R=1.0
    assert abs(by_sub["speech_clean"].f1 - 1.0) < 1e-9
    assert abs(by_sub["speech_clean"].precision - 1.0) < 1e-9
    assert abs(by_sub["speech_clean"].recall - 1.0) < 1e-9
    # music_pop: class 1 P=2/3 R=1 F1=4/5; class 2 P=0 R=0 F1=0
    # macro: F1=(4/5)/2=0.4, P=(2/3)/2=1/3, R=(1+0)/2=0.5
    assert abs(by_sub["music_pop"].f1 - 0.4) < 1e-9
    assert abs(by_sub["music_pop"].precision - 1/3) < 1e-9
    assert abs(by_sub["music_pop"].recall - 0.5) < 1e-9


def test_subclass_metrics_single_class():
    """Subclass with only one true class (e.g. noise — all inactive)."""
    y_true = np.array([2, 2, 2, 2])
    y_pred = np.array([2, 2, -1, 2])  # one frame wrong
    subs  = np.array(["noise"] * 4)
    by_sub = _compute_subclass_metrics(y_true, y_pred, subs)
    # present = [2]; single-class: P=3/3=1.0, R=3/4=0.75, F1=2*1*0.75/1.75=6/7
    assert abs(by_sub["noise"].precision - 1.0) < 1e-9
    assert abs(by_sub["noise"].recall - 0.75) < 1e-9
    assert abs(by_sub["noise"].f1 - 6/7) < 1e-9
