"""Тесты `LogRegL2Model`."""
from __future__ import annotations

import numpy as np

from src.models.logreg import LogRegL2Model


def test_logreg_fits_separable():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = LogRegL2Model(C=1.0).fit(X, y)
    proba = m.predict(X)
    assert proba.shape == (200,)
    assert ((proba >= 0) & (proba <= 1)).all()
    auc_proxy = float(((proba > 0.5) == y).mean())
    assert auc_proxy > 0.9


def test_logreg_single_class_fallback():
    """Если в y все одинаковые — модель не падает, возвращает константу."""
    X = np.random.default_rng(0).normal(0, 1, (50, 3))
    y = np.zeros(50, dtype=int)
    m = LogRegL2Model().fit(X, y)
    proba = m.predict(X)
    assert proba.shape == (50,)
    assert np.all(proba == 0.0)


def test_logreg_returns_probabilities_not_logits():
    X = np.random.default_rng(0).normal(0, 1, (100, 5))
    y = (X[:, 0] > 0).astype(int)
    m = LogRegL2Model().fit(X, y)
    p = m.predict(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
