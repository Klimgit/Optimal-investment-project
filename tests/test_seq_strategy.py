"""Тесты sequence-режима в `MLScoringStrategy`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pml.strategies.optimization_data import PredictionData, TrainingData

from src.backtest.strategy import MLScoringStrategy
from src.models.lstm import LSTMRegressor


def _make_panel(tmp: Path, n_months: int = 36, n_tickers: int = 12) -> Path:
    """Длинный panel с непрерывной историей по всем тикерам."""
    rng = np.random.default_rng(0)
    months = pd.bdate_range("2008-01-01", periods=n_months, freq="ME")
    tickers = [f"S{i:02d}" for i in range(n_tickers)]

    rows = []
    for d in months:
        for t in tickers:
            x = rng.normal(0, 1, 4)
            y = 0.4 * x[0] - 0.2 * x[1] + 0.05 * rng.normal()
            rows.append({
                "date": d, "ticker": t,
                "f0": x[0], "f1": x[1], "f2": x[2], "f3": x[3],
                "ret_next": y, "target_reg": y, "target_clf": float(y > 0),
            })
    p = tmp / "panel.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_seq_slice_train_shape(tmp_path):
    panel_p = _make_panel(tmp_path, n_months=36, n_tickers=12)
    strat = MLScoringStrategy(
        model_factory=lambda: None,
        panel_path=panel_p,
        train_window_months=24,
        sequence_length=6,
    )
    pred_date = pd.Timestamp("2010-06-30")
    slc = strat._slice_train(pred_date)
    assert slc.X.ndim == 3
    assert slc.X.shape[1] == 6
    assert slc.X.shape[2] == 4
    assert slc.X.shape[0] == slc.y.shape[0]
    assert slc.X.shape[0] > 0


def test_seq_strategy_full_roundtrip(tmp_path):
    panel_p = _make_panel(tmp_path, n_months=36, n_tickers=12)
    strat = MLScoringStrategy(
        model_factory=lambda: LSTMRegressor(
            hidden=8, epochs=5, batch_size=32, lr=5e-3, patience=2, val_frac=0.2, seed=0,
        ),
        panel_path=panel_p,
        train_window_months=24,
        sequence_length=6,
    )
    strat.universe = [f"S{i:02d}" for i in range(12)]

    pred_date = pd.Timestamp("2010-06-30")
    strat._fit(TrainingData(pred_date=pred_date))
    assert strat._fitted_model is not None

    scores = strat.predict_scores(PredictionData(pred_date=pred_date))
    assert isinstance(scores, pd.Series)
    assert scores.notna().all()
    assert set(scores.index).issubset(set(strat.universe))


def test_seq_strategy_skips_assets_without_full_history(tmp_path):
    """Если у ассета нет полной T=12 истории на pred_date, его пропустит."""
    panel_p = _make_panel(tmp_path, n_months=8, n_tickers=10)
    strat = MLScoringStrategy(
        model_factory=lambda: None,
        panel_path=panel_p,
        train_window_months=12,
        sequence_length=12,
    )
    pred_date = pd.Timestamp("2010-12-31")
    slc = strat._slice_train(pred_date)
    assert slc.X.shape[0] == 0
