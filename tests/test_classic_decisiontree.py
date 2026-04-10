# tests/test_classic_decisiontree.py

import numpy as np
import pytest
from src.classic.decisiontree import DecisionTree

VALID_LABELS = {-1, 1, 2}


def _train_dt(name="dt_test", n=210, n_features=40):
    rng = np.random.default_rng(0)
    # balanced 3-class data
    y = np.array([-1] * 70 + [1] * 70 + [2] * 70)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    dt = DecisionTree(name=name)
    dt._train(X, y)
    return dt, X, y


# ── construction validation ──────────────────────────────────────────────

def test_invalid_n_last_decisions_raises():
    with pytest.raises(ValueError):
        DecisionTree(name="x", n_last_decisions=0)


def test_invalid_forget_factor_zero_raises():
    with pytest.raises(ValueError):
        DecisionTree(name="x", decision_forget_factor=0.0)


def test_invalid_forget_factor_above_one_raises():
    with pytest.raises(ValueError):
        DecisionTree(name="x", decision_forget_factor=1.1)


# ── smoothing weights ────────────────────────────────────────────────────

def test_smoothing_weights_sum_to_one():
    dt = DecisionTree(name="x", n_last_decisions=30)
    assert abs(dt.smoothing_weights.sum() - 1.0) < 1e-6


def test_smoothing_weights_monotone_decrease():
    dt = DecisionTree(name="x", n_last_decisions=10)
    w = dt.smoothing_weights
    # weights[0] corresponds to most recent → should be largest
    assert all(w[i] >= w[i + 1] for i in range(len(w) - 1))


def test_smoothing_weights_length():
    n = 15
    dt = DecisionTree(name="x", n_last_decisions=n)
    assert len(dt.smoothing_weights) == n


# ── predict ──────────────────────────────────────────────────────────────

def test_predict_returns_valid_label(classic_config_dt):
    dt, X, y = _train_dt()
    result = dt.predict(X[0])
    assert result in VALID_LABELS


def test_predict_accumulates_decisions(classic_config_dt):
    dt, X, y = _train_dt()
    for i in range(5):
        dt.predict(X[i])
    assert len(dt.last_decisions) == 5


# ── predict_batch ────────────────────────────────────────────────────────

def test_predict_batch_shape(classic_config_dt):
    dt, X, y = _train_dt()
    preds = dt.predict_batch(X[:20])
    assert preds.shape == (20,)


def test_predict_batch_valid_labels(classic_config_dt):
    dt, X, y = _train_dt()
    preds = dt.predict_batch(X[:30])
    assert set(preds).issubset(VALID_LABELS)


# ── predict_proba ────────────────────────────────────────────────────────

def test_predict_proba_shape(classic_config_dt):
    dt, X, y = _train_dt()
    proba = dt.predict_proba(X[0])
    assert proba.ndim == 1
    assert len(proba) == 3


def test_predict_proba_sums_to_one(classic_config_dt):
    dt, X, y = _train_dt()
    proba = dt.predict_proba(X[0])
    assert abs(proba.sum() - 1.0) < 1e-5
