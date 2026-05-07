"""Построение universe на каждый конец месяца.

Фильтры (по умолчанию из `configs/base.yaml`):
- история ≥ `min_history_months` месяцев на дату снимка;
- цена закрытия > `min_price` (отсекаем penny stocks);
- 21-дневный средний dollar-volume рассчитан (т.е. достаточно данных);
- тикер «жив» в следующем баре (есть данные после даты снимка) — простой
  прокси на отсутствие делистинга в момент ребалансировки.

Из оставшихся берём Top-N по 21d ADV (average dollar volume).

Запуск как модуль:

    python -m src.data.universe [--config base.yaml]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.utils.io import load_config, read_parquet, write_parquet

logger = logging.getLogger(__name__)


def compute_adv(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Добавить столбец `adv` — скользящее среднее `close*volume` за `lookback` дней.

    Возвращает копию `prices` с дополнительными столбцами `dv` (daily) и `adv`.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    df["dv"] = df["close"].astype("float64") * df["volume"].astype("float64")
    df["adv"] = (
        df.groupby("ticker", observed=True)["dv"]
        .transform(lambda s: s.rolling(lookback, min_periods=lookback).mean())
    )
    return df


def month_end_trading_days(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Последний торговый день каждого календарного месяца, присутствующего в `dates`."""
    s = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).unique())).sort_values()
    period = s.to_period("M")
    df = pd.DataFrame({"date": s, "ym": period})
    last = df.groupby("ym", as_index=False)["date"].last()["date"]
    return pd.DatetimeIndex(last.sort_values().values)


def build_universe(
    prices: pd.DataFrame,
    top_n: int = 1500,
    min_history_months: int = 24,
    min_price: float = 5.0,
    adv_lookback: int = 21,
) -> pd.DataFrame:
    """На каждый конец месяца выбрать Top-N тикеров по 21d ADV с фильтрами.

    Returns
    -------
    DataFrame `[date, ticker]` — ребаланс-даты и активные тикеры.
    """
    df = compute_adv(prices, lookback=adv_lookback)

    grp = df.groupby("ticker", observed=True)["date"]
    df["first_seen"] = grp.transform("min")
    df["last_seen"] = grp.transform("max")

    month_ends = month_end_trading_days(df["date"])
    snap = df[df["date"].isin(month_ends)].copy()

    cutoff_first = snap["date"] - pd.DateOffset(months=min_history_months)
    snap = snap[
        snap["adv"].notna()
        & (snap["close"] >= min_price)
        & (snap["first_seen"] <= cutoff_first)
        & (snap["last_seen"] > snap["date"])
    ]

    out = (
        snap.sort_values(["date", "adv"], ascending=[True, False])
        .groupby("date", group_keys=False, observed=True)
        .head(top_n)
        .loc[:, ["date", "ticker"]]
        .reset_index(drop=True)
    )
    out["ticker"] = out["ticker"].astype(str)

    logger.info(
        "Universe: %d snapshots, %d total (date,ticker) pairs, mean=%.0f tickers/month",
        out["date"].nunique(),
        len(out),
        len(out) / max(out["date"].nunique(), 1),
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build monthly universe from prices.parquet")
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--out", default=None, help="override output parquet path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prices_path = cfg["paths"]["processed_prices"]
    if not Path(prices_path).exists():
        raise FileNotFoundError(
            f"{prices_path} не найден; сначала запустите `python -m src.data.loader`"
        )

    prices = read_parquet(prices_path)
    universe = build_universe(
        prices,
        top_n=cfg["universe"]["top_n"],
        min_history_months=cfg["universe"]["min_history_months"],
        min_price=cfg["universe"]["min_price"],
        adv_lookback=cfg["rebalance"]["lag_trading_days"],
    )

    out_path = args.out or "data/processed/universe.parquet"
    write_parquet(universe, out_path)
    logger.info("Wrote %d rows to %s", len(universe), out_path)


if __name__ == "__main__":
    main()
