"""Тесты MLPRegressor и LSTMRegressor (smoke + numerical)."""
from __future__ import annotations

import numpy as np

from src.models.lstm import LSTMRegressor
from src.models.mlp import MLPRegressor


def _linear_xy(n: int = 600, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    y = (1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def test_mlp_fits_simple_signal():
    X, y = _linear_xy()
    m = MLPRegressor(hidden=(16, 8), dropout=0.1, epochs=20, batch_size=64, lr=5e-3, seed=0)
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (X.shape[0],)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.7


def test_mlp_rejects_3d_input():
    X = np.zeros((10, 3, 4), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    m = MLPRegressor(epochs=1, batch_size=4, val_frac=0.2)
    try:
        m.fit(X, y)
    except ValueError:
        return
    raise AssertionError("MLP должен падать на 3D-входе")


def _seq_xy(n: int = 300, T: int = 6, d: int = 4, seed: int = 0):
    """Сигнал на последнем шаге: y = f(X[:, -1, :])."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, T, d)).astype(np.float32)
    y = (X[:, -1, 0] - 0.5 * X[:, -2, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def test_lstm_fits_sequence_signal():
    X, y = _seq_xy()
    m = LSTMRegressor(hidden=16, num_layers=1, dropout=0.0, epochs=25, batch_size=64, lr=5e-3, patience=5, seed=0)
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (X.shape[0],)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.4


def test_lstm_rejects_2d_input():
    X = np.zeros((10, 4), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    m = LSTMRegressor(epochs=1, val_frac=0.2)
    try:
        m.fit(X, y)
    except ValueError:
        return
    raise AssertionError("LSTM должен падать на 2D-входе")
