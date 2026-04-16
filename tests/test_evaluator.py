# tests/test_evaluator.py

import numpy as np
import pytest
from src.evaluator import (
    run_evaluation,
    format_report,
    _compute_subclass_metrics,
    _bootstrap_ci,
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


def test_subclass_metrics_primary_class():
    """speech_clean (primary=-1) and music_pop (primary=1): binary-on-mask F1/P/R."""
    # speech_clean: 2 speech + 1 inactive frame; music_pop: 2 music + 1 inactive frame
    y_true = np.array([-1, -1,  2,  1,  1,  2])
    y_pred = np.array([-1, -1,  2,  1,  1,  1])  # music_pop: inactive frame mispredicted as music
    subs  = np.array(["speech_clean"] * 3 + ["music_pop"] * 3)
    by_sub = _compute_subclass_metrics(y_true, y_pred, subs)
    # speech_clean (primary=-1): y_t_bin=[1,1,0], y_p_bin=[1,1,0] → perfect
    assert abs(by_sub["speech_clean"].f1 - 1.0) < 1e-9
    assert abs(by_sub["speech_clean"].precision - 1.0) < 1e-9
    assert abs(by_sub["speech_clean"].recall - 1.0) < 1e-9
    # music_pop (primary=1): y_t_bin=[1,1,0], y_p_bin=[1,1,1] → TP=2, FP=1, FN=0
    # P = 2/3, R = 1.0, F1 = 2*(2/3)*1 / (2/3 + 1) = (4/3) / (5/3) = 4/5
    assert abs(by_sub["music_pop"].precision - 2/3) < 1e-9
    assert abs(by_sub["music_pop"].recall - 1.0) < 1e-9
    assert abs(by_sub["music_pop"].f1 - 4/5) < 1e-9


def test_subclass_metrics_noise_only():
    """noise (primary=2): binary-on-mask degenerates to single-class accuracy."""
    y_true = np.array([2, 2, 2, 2])
    y_pred = np.array([2, 2, -1, 2])  # one frame wrong
    subs  = np.array(["noise"] * 4)
    by_sub = _compute_subclass_metrics(y_true, y_pred, subs)
    # y_t_bin=[1,1,1,1], y_p_bin=[1,1,0,1] → TP=3, FP=0, FN=1 → P=1.0, R=0.75, F1=6/7
    assert abs(by_sub["noise"].precision - 1.0) < 1e-9
    assert abs(by_sub["noise"].recall - 0.75) < 1e-9
    assert abs(by_sub["noise"].f1 - 6/7) < 1e-9


def test_weighted_f1_computed():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 0.0, "weighted", save_to_file=False)
    assert abs(res.weighted_f1 - 1.0) < 1e-6  # perfect predictions


def test_accuracy_in_report():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 0.0, "acc_rep", save_to_file=False)
    assert abs(res.accuracy - 1.0) < 1e-6
    assert "Accuracy" in format_report(res)


def test_macro_auroc_present_with_scores():
    y_true, subs = _make_arrays(99)
    # Perfect scores: one-hot at true label
    n = len(y_true)
    y_scores = np.zeros((n, 3), dtype=np.float32)
    label_to_col = {-1: 0, 1: 1, 2: 2}
    for i, l in enumerate(y_true):
        y_scores[i, label_to_col[int(l)]] = 1.0
    res = run_evaluation(y_true, y_true.copy(), subs, 0.0, "auroc", save_to_file=False, y_scores=y_scores)
    assert res.macro_auroc is not None
    assert abs(res.macro_auroc - 1.0) < 1e-6


def test_macro_auroc_none_without_scores():
    y_true, subs = _make_arrays(99)
    res = run_evaluation(y_true, y_true.copy(), subs, 0.0, "no_auroc", save_to_file=False)
    assert res.macro_auroc is None


def test_bootstrap_ci_bounds_and_determinism():
    # 60 frames, 6 clips of 10 frames each, perfect predictions
    y_true = np.tile([-1, 1, 2], 20)
    y_pred = y_true.copy()
    clip_ids = np.repeat(np.arange(6), 10)
    labels = [-1, 1, 2]
    macro_ci_1, per_ci_1 = _bootstrap_ci(y_true, y_pred, clip_ids, labels, n_draws=200)
    macro_ci_2, per_ci_2 = _bootstrap_ci(y_true, y_pred, clip_ids, labels, n_draws=200)
    # Deterministic
    assert macro_ci_1 == macro_ci_2
    assert per_ci_1 == per_ci_2
    # Bounds well-ordered and contain point estimate (= 1.0 for perfect preds)
    assert macro_ci_1[0] <= 1.0 <= macro_ci_1[1] + 1e-9
    assert 0.0 <= macro_ci_1[0] <= macro_ci_1[1] <= 1.0 + 1e-9


def test_bootstrap_ci_via_run_evaluation():
    y_true = np.tile([-1, 1, 2], 10)
    y_pred = y_true.copy()
    subs = np.tile(["speech_clean", "music_pop", "noise"], 10)
    clip_ids = np.repeat(np.arange(6), 5)
    res = run_evaluation(
        y_true, y_pred, subs, 0.0, "ci_flow",
        save_to_file=False, clip_ids=clip_ids,
    )
    assert res.macro_f1_ci is not None
    assert res.per_class[-1].f1_ci is not None
    report = format_report(res)
    assert "[" in report and "]" in report  # CI rendered in report
