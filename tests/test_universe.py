"""Тесты universe.py на синтетических ценах."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.universe import build_universe, compute_adv, month_end_trading_days


def _make_synth_prices(
    tickers: list[str],
    start: str = "2008-01-01",
    end: str = "2012-06-30",
    base_price: float = 50.0,
    base_volume: int = 1_000_000,
) -> pd.DataFrame:
    """Сгенерировать ровные daily-цены: одинаковые для всех тикеров."""
    dates = pd.bdate_range(start=start, end=end)
    rows = []
    for t in tickers:
        rows.append(pd.DataFrame({
            "date": dates,
            "ticker": t,
            "open": base_price,
            "high": base_price * 1.01,
            "low": base_price * 0.99,
            "close": base_price,
            "volume": base_volume,
        }))
    return pd.concat(rows, ignore_index=True)


def test_month_end_trading_days_takes_last_business_day():
    dates = pd.bdate_range("2010-01-01", "2010-03-31")
    me = month_end_trading_days(dates)
    assert len(me) == 3
    assert me[0] == pd.Timestamp("2010-01-29")
    assert me[1] == pd.Timestamp("2010-02-26")
    assert me[2] == pd.Timestamp("2010-03-31")


def test_compute_adv_rolling_mean():
    prices = _make_synth_prices(["AAA"], start="2010-01-04", end="2010-02-12")
    df = compute_adv(prices, lookback=21)
    assert "adv" in df.columns
    assert df["adv"].iloc[:20].isna().all()
    assert np.isclose(df["adv"].iloc[20], 50.0 * 1_000_000)


def test_build_universe_top_n_by_dollar_volume():
    """Тикеры с большим volume должны попасть в топ. История у всех одинакова."""
    tickers = [f"T{i:03d}" for i in range(10)]
    prices = _make_synth_prices(tickers, start="2008-01-01", end="2012-06-30")
    bumps = {t: i + 1 for i, t in enumerate(tickers)}
    prices["volume"] = prices["volume"] * prices["ticker"].map(bumps).astype(int)

    uni = build_universe(prices, top_n=3, min_history_months=24, min_price=5.0,
                         adv_lookback=21)
    assert (uni["date"].nunique() > 0)
    last_date = uni["date"].max()
    last_set = sorted(uni[uni["date"] == last_date]["ticker"].tolist())
    assert last_set == ["T007", "T008", "T009"]


def test_build_universe_filters_short_history():
    """Тикеры с историей < 24 мес не должны попадать в universe."""
    long_t = _make_synth_prices(["LONG"], start="2008-01-01", end="2012-06-30")
    short_t = _make_synth_prices(["SHRT"], start="2011-06-01", end="2012-06-30")
    prices = pd.concat([long_t, short_t], ignore_index=True)

    uni = build_universe(prices, top_n=10, min_history_months=24, min_price=5.0,
                         adv_lookback=21)
    early = uni[uni["date"] <= "2010-12-31"]
    assert (early["ticker"] == "LONG").all()
    assert "SHRT" not in early["ticker"].unique().tolist()


def test_build_universe_filters_penny_stocks():
    cheap = _make_synth_prices(["CHEAP"], base_price=2.0)
    ok = _make_synth_prices(["OK"], base_price=50.0)
    prices = pd.concat([cheap, ok], ignore_index=True)

    uni = build_universe(prices, top_n=10, min_history_months=24, min_price=5.0,
                         adv_lookback=21)
    assert "CHEAP" not in uni["ticker"].unique().tolist()
    assert "OK" in uni["ticker"].unique().tolist()


def test_build_universe_drops_dead_tickers_at_last_snapshot():
    """Тикер, у которого нет данных после месяца снимка, не попадает (proxy на делистинг)."""
    alive = _make_synth_prices(["ALIVE"], start="2008-01-01", end="2012-06-30")
    dead = _make_synth_prices(["DEAD"], start="2008-01-01", end="2010-12-31")
    prices = pd.concat([alive, dead], ignore_index=True)

    uni = build_universe(prices, top_n=10, min_history_months=24, min_price=5.0,
                         adv_lookback=21)
    last_dates_by_ticker = uni.groupby("ticker", observed=True)["date"].max()
    assert last_dates_by_ticker["DEAD"] < pd.Timestamp("2011-01-01")
    assert last_dates_by_ticker["ALIVE"] >= pd.Timestamp("2012-05-01")
