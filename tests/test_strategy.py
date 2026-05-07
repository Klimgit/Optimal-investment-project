"""Тесты для `MLScoringStrategy` и `RidgeModel` на синтетике."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pml.strategies.optimization_data import PredictionData, TrainingData

from src.backtest.strategy import MLScoringStrategy
from src.models.ridge import RidgeModel


def _make_synthetic_panel(tmp: Path, n_months: int = 72, n_tickers: int = 30) -> Path:
    """Сгенерировать panel.parquet: 5 фич, target_reg/target_clf/ret_next."""
    rng = np.random.default_rng(42)
    months = pd.bdate_range("2008-01-01", periods=n_months, freq="ME")
    tickers = [f"S{i:03d}" for i in range(n_tickers)]

    rows = []
    for d in months:
        for t in tickers:
            x = rng.normal(0, 1, 5)
            y = 0.3 * x[0] - 0.2 * x[1] + 0.05 * rng.normal()
            rows.append({
                "date": d,
                "ticker": t,
                "f0": x[0], "f1": x[1], "f2": x[2], "f3": x[3], "f4": x[4],
                "ret_next": y,
                "target_reg": y - 0.0,
                "target_clf": float(y > np.quantile([0.0], 0.9)),
            })
    panel = pd.DataFrame(rows)
    p = tmp / "panel.parquet"
    panel.to_parquet(p, index=False)
    return p


def test_ridge_fits_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 5))
    y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=200)
    m = RidgeModel(alpha=0.1).fit(X, y)
    pred = m.predict(X)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.9
    assert pred.shape == (200,)


def test_strategy_fit_predict_roundtrip(tmp_path):
    panel_p = _make_synthetic_panel(tmp_path, n_months=72, n_tickers=30)

    strat = MLScoringStrategy(
        model_factory=lambda: RidgeModel(alpha=0.1),
        panel_path=panel_p,
        target_col="target_reg",
        train_window_months=36,
        mode="long_short",
        quantile=0.1,
    )
    strat.universe = [f"S{i:03d}" for i in range(30)]

    pred_date = pd.Timestamp("2012-12-31")
    train = TrainingData(pred_date=pred_date)
    strat._fit(train)
    assert strat._fitted_model is not None

    pred = PredictionData(pred_date=pred_date)
    scores = strat.predict_scores(pred)
    assert isinstance(scores, pd.Series)
    assert scores.notna().all()
    assert set(scores.index).issubset(set(strat.universe))


def test_strategy_uses_only_train_window(tmp_path):
    """`_fit` должен видеть только данные строго ДО pred_date."""
    panel_p = _make_synthetic_panel(tmp_path, n_months=60, n_tickers=20)
    strat = MLScoringStrategy(
        model_factory=lambda: RidgeModel(),
        panel_path=panel_p,
        train_window_months=24,
    )

    pred_date = pd.Timestamp("2011-06-30")
    slc = strat._slice_train(pred_date)
    panel = strat._load_panel()
    used_dates = panel.index.get_level_values("date").unique()

    assert slc.n_snapshots > 0
    assert slc.n_rows > 0
    assert slc.n_snapshots <= 24


def test_strategy_no_train_data_returns_empty(tmp_path):
    """Если до pred_date < 100 строк — не учим модель, возвращаем пусто."""
    panel_p = _make_synthetic_panel(tmp_path, n_months=72, n_tickers=30)
    strat = MLScoringStrategy(
        model_factory=lambda: RidgeModel(),
        panel_path=panel_p,
        train_window_months=60,
    )
    strat.universe = [f"S{i:03d}" for i in range(30)]

    pred_date = pd.Timestamp("2007-01-01")
    train = TrainingData(pred_date=pred_date)
    strat._fit(train)
    assert strat._fitted_model is None

    pred = PredictionData(pred_date=pred_date)
    scores = strat.predict_scores(pred)
    assert len(scores) == 0
