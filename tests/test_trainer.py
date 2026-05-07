"""Тесты для общего torch-trainer."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.training.trainer import TrainConfig, predict_torch, train_torch_regressor


def _make_linear_data(n: int = 500, d: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    w = rng.normal(0, 1, d).astype(np.float32)
    y = X @ w + 0.1 * rng.normal(size=n).astype(np.float32)
    return X, y


def test_trainer_reduces_loss_on_linear_problem():
    X, y = _make_linear_data()
    model = nn.Linear(X.shape[1], 1)
    cfg = TrainConfig(epochs=15, batch_size=64, lr=1e-2, val_frac=0.2, patience=3, seed=0)
    res = train_torch_regressor(model, X, y, cfg)
    assert res.epochs_run > 0
    assert res.train_losses[-1] < res.train_losses[0]
    assert res.best_val_loss < float("inf")
    pred = predict_torch(model, X)
    assert pred.shape == (X.shape[0],)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.9


def test_trainer_early_stopping_kicks_in():
    """Если данные шумные и patience=1 — early stop должен сработать раньше epochs."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3)).astype(np.float32)
    y = rng.normal(0, 1, 200).astype(np.float32)  # pure noise

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 1)

        def forward(self, x):
            return self.fc(x)

    model = Tiny()
    cfg = TrainConfig(epochs=200, batch_size=64, lr=1e-2, val_frac=0.3, patience=2, seed=0)
    res = train_torch_regressor(model, X, y, cfg)
    assert res.epochs_run < cfg.epochs


def test_trainer_seeded_reproducible():
    X, y = _make_linear_data()
    cfg = TrainConfig(epochs=10, batch_size=64, lr=1e-2, val_frac=0.2, patience=5, seed=123)

    torch.manual_seed(0)
    m1 = nn.Linear(X.shape[1], 1)
    train_torch_regressor(m1, X, y, cfg)

    torch.manual_seed(0)
    m2 = nn.Linear(X.shape[1], 1)
    train_torch_regressor(m2, X, y, cfg)

    p1 = predict_torch(m1, X)
    p2 = predict_torch(m2, X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)
