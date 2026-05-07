"""Smoke-тесты графиков: что они вообще строятся без ошибок."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd

from src.evaluation.plots import (
    plot_comparison,
    plot_drawdown,
    plot_equity,
    plot_rolling_sharpe,
    plot_weights_distribution,
)


def _fake_returns(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=idx, name="strategy")


def test_plot_equity_runs():
    s = _fake_returns()
    b = _fake_returns(seed=1)
    fig = plot_equity(s, b, strategy_name="test", benchmark_name="bench")
    assert fig is not None


def test_plot_drawdown_runs():
    fig = plot_drawdown(_fake_returns(), strategy_name="test")
    assert fig is not None


def test_plot_rolling_sharpe_short_window():
    fig = plot_rolling_sharpe(_fake_returns(), window_days=63, strategy_name="test")
    assert fig is not None


def test_plot_weights_distribution_runs():
    rng = np.random.default_rng(0)
    n_dates, n_assets = 50, 20
    idx = pd.date_range("2015-01-01", periods=n_dates, freq="ME")
    w = rng.normal(0, 0.05, (n_dates, n_assets))
    df = pd.DataFrame(w, index=idx, columns=[f"T{i}" for i in range(n_assets)])
    fig = plot_weights_distribution(df, strategy_name="test")
    assert fig is not None


def test_plot_comparison_runs():
    strategies = {f"s{i}": _fake_returns(seed=i) for i in range(3)}
    bench = _fake_returns(seed=99)
    fig = plot_comparison(strategies, bench)
    assert fig is not None
