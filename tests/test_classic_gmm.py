# tests/test_classic_gmm.py

import numpy as np
import pytest
from src.classic.gmm import GMM

VALID_LABELS = {-1, 1, 2}
N_FEATURES = 9


def _train_gmm(name="gmm_test", n_per_class=30):
    rng = np.random.default_rng(0)
    n = n_per_class * 3
    y = np.array([-1] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float64)
    gmm = GMM(name=name)
    gmm._train(X, y)
    return gmm, X, y


# ── training validation ───────────────────────────────────────────────────

def test_missing_class_raises(classic_config_gmm):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, N_FEATURES))
    y = np.array([-1] * 30 + [1] * 30)  # no inactive class
    gmm = GMM(name="x")
    with pytest.raises(ValueError, match="inactive"):
        gmm._train(X, y)


# ── predict ──────────────────────────────────────────────────────────────

def test_predict_returns_valid_label(classic_config_gmm):
    gmm, X, y = _train_gmm()
    result = gmm.predict(X[0])
    assert result in VALID_LABELS


def test_predict_uses_smoothing_buffer(classic_config_gmm):
    gmm, X, y = _train_gmm()
    assert len(gmm.ll_buf) == 0
    gmm.predict(X[0])
    assert len(gmm.ll_buf) == 1


def test_smoothing_buffer_maxlen(classic_config_gmm):
    gmm, X, y = _train_gmm()
    assert gmm.ll_buf.maxlen == 66


# ── predict_proba ────────────────────────────────────────────────────────

def test_predict_proba_sums_to_one(classic_config_gmm):
    gmm, X, y = _train_gmm()
    proba = gmm.predict_proba(X[0])
    assert abs(proba.sum() - 1.0) < 1e-5


def test_predict_proba_non_negative(classic_config_gmm):
    gmm, X, y = _train_gmm()
    proba = gmm.predict_proba(X[0])
    assert (proba >= 0).all()


def test_predict_proba_length(classic_config_gmm):
    gmm, X, y = _train_gmm()
    proba = gmm.predict_proba(X[0])
    assert len(proba) == 3


# ── predict_batch ────────────────────────────────────────────────────────

def test_predict_batch_shape(classic_config_gmm):
    gmm, X, y = _train_gmm()
    preds = gmm.predict_batch(X[:20])
    assert preds.shape == (20,)


def test_predict_batch_valid_labels(classic_config_gmm):
    gmm, X, y = _train_gmm()
    preds = gmm.predict_batch(X[:30])
    assert set(preds).issubset(VALID_LABELS)


# ── predict_proba_batch ──────────────────────────────────────────────────

def test_predict_proba_batch_sums_to_one(classic_config_gmm):
    gmm, X, y = _train_gmm()
    proba = gmm.predict_proba_batch(X[:10])
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
