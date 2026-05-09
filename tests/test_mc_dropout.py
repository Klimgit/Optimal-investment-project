"""Тесты MC-Dropout MLP-классификатора и uncertainty-стратегии."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pml.strategies.optimization_data import PredictionData, TrainingData

from src.backtest.mc_strategy import MCDropoutScoringStrategy
from src.models.mc_dropout_mlp import MCDropoutMLPClassifier


def _binary_xy(n: int = 600, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    logits = 1.5 * X[:, 0] - 0.7 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return X, y


def test_mc_dropout_fits_simple_signal():
    X, y = _binary_xy()
    m = MCDropoutMLPClassifier(
        hidden=(16, 8), dropout=0.5, n_mc_samples=10,
        epochs=15, batch_size=64, lr=5e-3, seed=0,
    )
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (X.shape[0],)
    assert ((p >= 0) & (p <= 1)).all()
    auc_proxy = float(((p > 0.5) == y).mean())
    assert auc_proxy > 0.65


def test_mc_dropout_returns_uncertainty():
    X, y = _binary_xy(n=200)
    m = MCDropoutMLPClassifier(
        hidden=(8, 4), dropout=0.5, n_mc_samples=20,
        epochs=5, batch_size=32, lr=5e-3, seed=0,
    )
    m.fit(X, y)
    proba, sigma = m.predict_with_uncertainty(X)
    assert proba.shape == (200,)
    assert sigma.shape == (200,)
    assert (sigma >= 0).all()
    assert sigma.std() > 1e-6                                   


def test_mc_dropout_single_class_fallback():
    """Если y одинаковый — модель не учится, возвращает константу."""
    X = np.random.default_rng(0).normal(0, 1, (50, 3)).astype(np.float32)
    y = np.ones(50, dtype=np.float32)
    m = MCDropoutMLPClassifier(epochs=1, batch_size=8, n_mc_samples=5)
    m.fit(X, y)
    p = m.predict(X)
    np.testing.assert_allclose(p, 1.0)


def _make_clf_panel(tmp: Path, n_months: int = 36, n_tickers: int = 30) -> Path:
    rng = np.random.default_rng(0)
    months = pd.bdate_range("2008-01-01", periods=n_months, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    rows = []
    for d in months:
        for t in tickers:
            x = rng.normal(0, 1, 4)
            logit = 1.0 * x[0] - 0.5 * x[1]
            p = 1.0 / (1.0 + np.exp(-logit))
            y_clf = float(rng.uniform() < p)
            rows.append({
                "date": d, "ticker": t,
                "f0": x[0], "f1": x[1], "f2": x[2], "f3": x[3],
                "ret_next": logit, "target_reg": logit, "target_clf": y_clf,
            })
    p = tmp / "panel.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_mc_strategy_with_uncertainty_filter(tmp_path):
    panel_p = _make_clf_panel(tmp_path)
    strat = MCDropoutScoringStrategy(
        model_factory=lambda: MCDropoutMLPClassifier(
            hidden=(8, 4), dropout=0.5, n_mc_samples=10,
            epochs=4, batch_size=32, lr=5e-3, seed=0,
        ),
        panel_path=panel_p,
        target_col="target_clf",
        train_window_months=24,
        sequence_length=1,
        uncertainty_quantile=0.5,
    )
    strat.universe = [f"T{i:03d}" for i in range(30)]

    pred_date = pd.Timestamp("2010-12-31")
    strat._fit(TrainingData(pred_date=pred_date))
    assert strat._fitted_model is not None

    scores = strat.predict_scores(PredictionData(pred_date=pred_date))
    assert isinstance(scores, pd.Series)
    assert scores.notna().all()
                                                                               
    assert len(scores) <= 30


def test_mc_strategy_falls_back_when_model_lacks_uncertainty(tmp_path):
    """Если модель не имеет predict_with_uncertainty — стратегия работает как обычная."""
    from src.models.ridge import RidgeModel

    panel_p = _make_clf_panel(tmp_path)
    strat = MCDropoutScoringStrategy(
        model_factory=lambda: RidgeModel(alpha=0.1),
        panel_path=panel_p,
        target_col="target_clf",
        train_window_months=24,
        uncertainty_quantile=0.5,
    )
    strat.universe = [f"T{i:03d}" for i in range(30)]

    pred_date = pd.Timestamp("2010-12-31")
    strat._fit(TrainingData(pred_date=pred_date))
    scores = strat.predict_scores(PredictionData(pred_date=pred_date))
    assert isinstance(scores, pd.Series)
    assert len(scores) > 0
