"""Smoke-тесты для `src/backtest/dataset_adapter.py` на синтетике."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.dataset_adapter import build_kaggle_dataset
from src.backtest.experiment_config import KaggleUSExperimentConfig


def _make_synthetic(tmp: Path, *, n_tickers: int = 5, n_days: int = 600) -> dict[str, Path]:
    """Сгенерировать прайсы / universe / spx в parquet-формате."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2008-01-02", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]

    rows = []
    for t in tickers:
        log_p = np.cumsum(rng.normal(0.0005, 0.02, n_days)) + np.log(50.0)
        close = np.exp(log_p)
        rows.append(pd.DataFrame({
            "date": days,
            "ticker": t,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e5, 1e7, n_days),
        }))
    prices = pd.concat(rows, ignore_index=True)
    prices_p = tmp / "prices.parquet"
    prices.to_parquet(prices_p, index=False)

    me = days[days.is_month_end | (days == days[-1])]
    uni = pd.DataFrame([
        {"date": d, "ticker": t} for d in me for t in tickers
    ])
    uni_p = tmp / "universe.parquet"
    uni.to_parquet(uni_p, index=False)

    spx_close = np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_days))) * 1500
    spx = pd.DataFrame({"date": days, "close": spx_close, "adj_close": spx_close})
    spx_p = tmp / "spx.parquet"
    spx.to_parquet(spx_p, index=False)

    return {"prices": prices_p, "universe": uni_p, "spx": spx_p}


def test_dataset_basic_shape(tmp_path):
    paths = _make_synthetic(tmp_path)
    cfg = KaggleUSExperimentConfig()
    cfg.DATA_PROCESSING_START_DATE = pd.Timestamp("2008-06-01")
    cfg.START_DATE = pd.Timestamp("2009-01-01")
    cfg.END_DATE = pd.Timestamp("2010-04-01")

    ds = build_kaggle_dataset(cfg, prices_path=paths["prices"], universe_path=paths["universe"], spx_path=paths["spx"])

    assert ds.data is not None
    assert ds.presence_matrix is not None
    assert cfg.RF_NAME in ds.data.columns
    assert "spx-rf" in ds.data.columns
    n_tickers = len([c for c in ds.data.columns if c.startswith("T")])
    assert n_tickers == 5
    assert ds.presence_matrix.shape[1] == 5


def test_dataset_no_nan_in_factors(tmp_path):
    """`spx-rf` должен быть без NaN — иначе Assessor падает на OLS."""
    paths = _make_synthetic(tmp_path)
    cfg = KaggleUSExperimentConfig()
    cfg.DATA_PROCESSING_START_DATE = pd.Timestamp("2008-06-01")
    cfg.END_DATE = pd.Timestamp("2010-04-01")

    ds = build_kaggle_dataset(cfg, prices_path=paths["prices"], universe_path=paths["universe"], spx_path=paths["spx"])
    assert not ds.data["spx-rf"].isna().any()
    assert (ds.data[cfg.RF_NAME] == 0.0).all()


def test_presence_matrix_is_binary(tmp_path):
    paths = _make_synthetic(tmp_path)
    cfg = KaggleUSExperimentConfig()
    cfg.DATA_PROCESSING_START_DATE = pd.Timestamp("2008-06-01")
    cfg.END_DATE = pd.Timestamp("2010-04-01")

    ds = build_kaggle_dataset(cfg, prices_path=paths["prices"], universe_path=paths["universe"], spx_path=paths["spx"])
    pm = ds.presence_matrix
    assert pm.dtypes.eq("int8").all()
    assert pm.isin([0, 1]).all().all()
    assert pm.sum().sum() > 0


def test_data_index_is_business_days(tmp_path):
    paths = _make_synthetic(tmp_path)
    cfg = KaggleUSExperimentConfig()
    cfg.DATA_PROCESSING_START_DATE = pd.Timestamp("2008-06-01")
    cfg.END_DATE = pd.Timestamp("2010-04-01")

    ds = build_kaggle_dataset(cfg, prices_path=paths["prices"], universe_path=paths["universe"], spx_path=paths["spx"])
    idx = ds.data.index
    assert isinstance(idx, pd.DatetimeIndex)
    assert (idx.dayofweek < 5).all()


def test_universe_restriction(tmp_path):
    """Тикеры, которых нет в universe.parquet, должны быть исключены из data."""
    paths = _make_synthetic(tmp_path, n_tickers=5)
    uni = pd.read_parquet(paths["universe"])
    uni = uni[uni["ticker"].isin(["T0", "T1", "T2"])]
    uni.to_parquet(paths["universe"], index=False)

    cfg = KaggleUSExperimentConfig()
    cfg.DATA_PROCESSING_START_DATE = pd.Timestamp("2008-06-01")
    cfg.END_DATE = pd.Timestamp("2010-04-01")

    ds = build_kaggle_dataset(cfg, prices_path=paths["prices"], universe_path=paths["universe"], spx_path=paths["spx"])
    n_tickers = len([c for c in ds.data.columns if c.startswith("T")])
    assert n_tickers == 3
