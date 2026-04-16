# tests/test_classic_svm.py


import numpy as np
import pytest
from src.classic.svm import SVM, _subsample_balanced

VALID_LABELS = {-1, 1, 2}
N_FEATURES = 9


def _train_svm(name="svm_test", n_per_class=60):
    """Train on small data to keep tests fast."""
    rng = np.random.default_rng(0)
    n = n_per_class * 3
    y = np.array([-1] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float64)
    svm = SVM(name=name)
    svm._train(X, y)
    return svm, X, y


# ── _subsample_balanced ───────────────────────────────────────────────────

def test_subsample_balanced_equal_counts():
    rng = np.random.default_rng(0)
    y = np.array([-1] * 100 + [1] * 80 + [2] * 60)
    X = rng.standard_normal((len(y), N_FEATURES))
    X_s, y_s = _subsample_balanced(X, y, max_per_class=50)
    for label in [-1, 1, 2]:
        assert np.sum(y_s == label) == 50


def test_subsample_balanced_respects_max():
    rng = np.random.default_rng(1)
    y = np.array([-1] * 200 + [1] * 200 + [2] * 200)
    X = rng.standard_normal((len(y), N_FEATURES))
    _, y_s = _subsample_balanced(X, y, max_per_class=100)
    for label in [-1, 1, 2]:
        assert np.sum(y_s == label) == 100


def test_subsample_balanced_smaller_than_max_kept():
    rng = np.random.default_rng(2)
    y = np.array([-1] * 20 + [1] * 20 + [2] * 20)
    X = rng.standard_normal((60, N_FEATURES))
    X_s, y_s = _subsample_balanced(X, y, max_per_class=50)
    # fewer than max → all kept
    for label in [-1, 1, 2]:
        assert np.sum(y_s == label) == 20


# ── predict ──────────────────────────────────────────────────────────────

def test_predict_returns_valid_label(classic_config_svm):
    svm, X, y = _train_svm()
    result = svm.predict(X[0])
    assert result in VALID_LABELS


def test_predict_uses_smoothing_buffer(classic_config_svm):
    svm, X, y = _train_svm()
    assert len(svm.dec_buf) == 0
    svm.predict(X[0])
    assert len(svm.dec_buf) == 1


def test_smoothing_buffer_maxlen(classic_config_svm):
    svm, X, y = _train_svm()
    assert svm.dec_buf.maxlen == 20


# ── predict_batch ────────────────────────────────────────────────────────

def test_predict_batch_shape(classic_config_svm):
    svm, X, y = _train_svm()
    preds = svm.predict_batch(X[:20])
    assert preds.shape == (20,)


def test_predict_batch_valid_labels(classic_config_svm):
    svm, X, y = _train_svm()
    preds = svm.predict_batch(X[:30])
    assert set(preds).issubset(VALID_LABELS)


# ── predict_proba ────────────────────────────────────────────────────────

def test_predict_proba_shape(classic_config_svm):
    svm, X, y = _train_svm()
    proba = svm.predict_proba(X[0])
    assert proba.shape == (3,)


def test_predict_proba_sums_to_one(classic_config_svm):
    svm, X, y = _train_svm()
    proba = svm.predict_proba(X[0])
    assert abs(proba.sum() - 1.0) < 1e-6


def test_predict_proba_non_negative(classic_config_svm):
    svm, X, y = _train_svm()
    proba = svm.predict_proba(X[0])
    assert (proba >= 0).all()


def test_predict_proba_batch_shape(classic_config_svm):
    svm, X, y = _train_svm()
    proba = svm.predict_proba_batch(X[:10])
    assert proba.shape == (10, 3)


def test_predict_proba_batch_sums_to_one(classic_config_svm):
    svm, X, y = _train_svm()
    proba = svm.predict_proba_batch(X[:10])
    assert np.allclose(proba.sum(axis=1), 1.0)
