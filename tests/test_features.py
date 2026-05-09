"""Численные тесты на синтетических ценах для features.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    MOMENTUM_COLS,
    REGIME_FEATURE_COLS,
    build_features,
    build_monthly_panel,
    compute_daily_features,
    compute_spx_regime_series,
    feature_columns,
    macd_pair_names,
    regime_row_asof,
    zscore_within_date,
)

PAIRS = [(8, 24), (16, 48)]


def _make_constant(ticker: str, n_days: int = 400, price: float = 50.0) -> pd.DataFrame:
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": price, "high": price, "low": price,
        "close": price, "volume": 1_000_000,
    })


def _make_geometric(ticker: str, n_days: int, daily_ret: float, p0: float = 50.0) -> pd.DataFrame:
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    closes = p0 * np.power(1 + daily_ret, np.arange(n_days))
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": 1_000_000,
    })


def test_feature_columns_has_24_for_8_pairs():
    pairs = [(8, 24), (16, 48), (32, 96), (64, 192), (12, 26), (5, 35), (10, 30), (20, 60)]
    cols = feature_columns(pairs)
    assert len(cols) == 24
    macd, sig = macd_pair_names(pairs)
    assert len(macd) == 8 and len(sig) == 8


def test_constant_price_zero_momentum_zero_macd():
    """На константной цене все momentum/MACD/Signal = 0, sigma = 0 → vol-norm = NaN."""
    df = _make_constant("FLAT", n_days=400)
    out = compute_daily_features(df, macd_pairs=PAIRS)

    tail = out.iloc[-1]
    for col in ["r1", "r3", "r6", "r12"]:
        assert tail[col] == pytest.approx(0.0, abs=1e-12)
    assert tail["sigma_ann"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(tail["r3_n"]) and np.isnan(tail["r6_n"]) and np.isnan(tail["r12_n"])

    for fast, slow in PAIRS:
        assert tail[f"macd_{fast}_{slow}"] == pytest.approx(0.0, abs=1e-12)
        assert tail[f"sig_{fast}_{slow}"] == pytest.approx(0.0, abs=1e-12)


def test_geometric_growth_matches_closed_form_returns():
    """На p_t = p_0 * (1+r)^t: r1 = (1+r)^21 - 1, r3 = (1+r)^63 - 1, etc."""
    daily_r = 0.001
    df = _make_geometric("UP", n_days=400, daily_ret=daily_r)
    out = compute_daily_features(df, macd_pairs=PAIRS)
    tail = out.iloc[-1]

    expected = {
        "r1": (1 + daily_r) ** 21 - 1,
        "r3": (1 + daily_r) ** 63 - 1,
        "r6": (1 + daily_r) ** 126 - 1,
        "r12": (1 + daily_r) ** 252 - 1,
    }
    for k, v in expected.items():
        assert tail[k] == pytest.approx(v, rel=1e-9)


def test_geometric_growth_macd_positive():
    """На монотонно растущей цене MACD и Signal должны быть > 0 (fast > slow EMA)."""
    df = _make_geometric("UP", n_days=400, daily_ret=0.001)
    out = compute_daily_features(df, macd_pairs=PAIRS)
    tail = out.iloc[-1]
    for fast, slow in PAIRS:
        assert tail[f"macd_{fast}_{slow}"] > 0
        assert tail[f"sig_{fast}_{slow}"] > 0


def test_groupby_no_leakage_between_tickers():
    """EMA и rolling не должны «протекать» между активами при groupby."""
    a = _make_geometric("AAA", n_days=400, daily_ret=0.001)
    b = _make_constant("BBB", n_days=400, price=10.0)
    df = pd.concat([a, b], ignore_index=True)

    out = compute_daily_features(df, macd_pairs=PAIRS)
    tail_a = out[out["ticker"] == "AAA"].iloc[-1]
    tail_b = out[out["ticker"] == "BBB"].iloc[-1]

    for fast, slow in PAIRS:
        assert tail_a[f"macd_{fast}_{slow}"] > 0
        assert tail_b[f"macd_{fast}_{slow}"] == pytest.approx(0.0, abs=1e-12)


def test_volatility_close_to_theoretical():
    """std дневных доходностей синтетического GBM должен сходиться к параметру."""
    rng = np.random.default_rng(0)
    n = 500
    sigma_d = 0.02
    daily_rets = rng.normal(0, sigma_d, size=n)
    closes = 100 * np.cumprod(1 + daily_rets)
    dates = pd.bdate_range("2010-01-04", periods=n)
    df = pd.DataFrame({
        "date": dates, "ticker": "GBM",
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": 1_000_000,
    })
    out = compute_daily_features(df, macd_pairs=PAIRS, vol_lookback_days=63)
    tail = out.iloc[-1]
    expected_sigma_ann = sigma_d * np.sqrt(252)
    assert tail["sigma_ann"] == pytest.approx(expected_sigma_ann, rel=0.20)


def test_build_monthly_panel_target_and_zscore():
    """Полный pipeline на 4 тикерах: проверяем target_reg ex-mean и Z-score."""
    np.random.seed(0)
    tickers, dfs = ["A", "B", "C", "D"], []
    for i, t in enumerate(tickers):
        dfs.append(_make_geometric(t, n_days=400, daily_ret=0.0005 * (i + 1)))
    prices = pd.concat(dfs, ignore_index=True)

    daily = compute_daily_features(prices, macd_pairs=PAIRS, vol_lookback_days=63)
    feat_cols = feature_columns(PAIRS)

    rebal_dates = pd.bdate_range("2011-01-31", periods=4, freq="BME")
    universe = pd.DataFrame({
        "date": np.repeat(rebal_dates, len(tickers)),
        "ticker": tickers * len(rebal_dates),
    })

    panel = build_monthly_panel(daily, universe, feature_cols=feat_cols)
    assert not panel.empty
    assert set(["date", "ticker", "ret_next", "target_reg", "target_clf"]).issubset(panel.columns)

    means_by_date = panel.groupby("date")["target_reg"].mean()
    for d, m in means_by_date.items():
        if not np.isnan(m):
            assert m == pytest.approx(0.0, abs=1e-12)

    panel_z = zscore_within_date(panel, feat_cols)
    for col in feat_cols:
        for d, g in panel_z.groupby("date"):
            vals = g[col].dropna().values
            if len(vals) >= 2 and np.std(vals) > 0:
                assert abs(vals.mean()) < 1e-9
                assert abs(vals.std(ddof=1) - 1.0) < 1e-6


def test_spx_regime_series_and_asof_are_causal():
    idx = pd.bdate_range("2010-01-04", periods=260)
    close = pd.Series(100 * np.linspace(1, 1.3, len(idx)), index=idx)
    regime = compute_spx_regime_series(close)
    assert list(regime.columns) == REGIME_FEATURE_COLS
    d = pd.Timestamp("2011-06-01")
    row = regime_row_asof(regime, d)
    assert np.isfinite(row["reg_spx_dd126"])
    assert row["reg_spx_dd126"].item() == pytest.approx(0.0, abs=1e-9)


def test_build_monthly_panel_broadcasts_regime_columns():
    tickers = ["A", "B"]
    dfs = [_make_geometric(t, n_days=500, daily_ret=0.0005) for t in tickers]
    prices = pd.concat(dfs, ignore_index=True)
    daily = compute_daily_features(prices, macd_pairs=PAIRS)
    fc = feature_columns(PAIRS)
    rebal = pd.DatetimeIndex([pd.Timestamp("2011-01-31"), pd.Timestamp("2011-02-28")])
    universe = pd.DataFrame({"date": np.repeat(rebal, 2), "ticker": tickers * len(rebal)})
    idx_d = pd.bdate_range("2010-06-01", periods=260)
    spx_px = pd.Series(100 + np.arange(len(idx_d), dtype=float) * 0.05, index=idx_d)
    regime = compute_spx_regime_series(spx_px)
    panel = build_monthly_panel(daily, universe, fc, regime_daily=regime)
    for col in REGIME_FEATURE_COLS:
        assert col in panel.columns
    assert (panel.groupby("date")["reg_spx_vol63"].transform("nunique") == 1).all()


def test_build_features_end_to_end():
    """Smoke-тест полного `build_features` с конфигом-словарём."""
    tickers = ["AAA", "BBB", "CCC"]
    dfs = [_make_geometric(t, n_days=350, daily_ret=0.0008 * (i + 1))
           for i, t in enumerate(tickers)]
    prices = pd.concat(dfs, ignore_index=True)

    rebal_dates = pd.bdate_range("2011-01-31", periods=3, freq="BME")
    universe = pd.DataFrame({
        "date": np.repeat(rebal_dates, len(tickers)),
        "ticker": tickers * len(rebal_dates),
    })

    cfg = {
        "features": {
            "macd_pairs": [[8, 24], [16, 48]],
            "signal_ema_days": 9,
            "vol_lookback_days": 63,
            "vol_annualization": 252,
        },
        "target": {
            "classification": {"top_q": 0.34, "bottom_q": 0.34},
        },
    }
    panel = build_features(prices, universe, cfg)
    assert not panel.empty
    assert "target_reg" in panel.columns
    assert panel["date"].nunique() == 3
