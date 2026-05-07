"""Адаптер: наш `data/processed/*.parquet` → `quant_pml.dataset.DatasetData`.

`Runner._prepare()` ожидает один широкий `data` DataFrame, в котором лежат:
- цены закрытия по всем тикерам universe (level series),
- колонка `RF_NAME` — risk-free rate (мы кладём 0),
- все колонки из `FACTORS` — _excess returns_ (для нас — 'spx-rf').

`presence_matrix` — daily 0/1 матрица: 1, если тикер «жив» в рамках текущего
месячного universe. Расширяем месячные снимки до daily через ffill между
ребалансами.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pml.dataset.dataset_data import DatasetData

from src.backtest.experiment_config import KaggleUSExperimentConfig

logger = logging.getLogger(__name__)


def _load_prices_wide(prices_path: str | Path) -> pd.DataFrame:
    """Long → wide pivot: rows=date, cols=ticker, values=close."""
    prices = pd.read_parquet(prices_path, columns=["date", "ticker", "close"])
    prices["ticker"] = prices["ticker"].astype(str)
    return prices.pivot_table(
        index="date", columns="ticker", values="close",
        aggfunc="last", observed=True,
    ).sort_index()


def _load_spx_excess(spx_path: str | Path) -> pd.Series:
    """Daily simple return ^GSPC. RF=0, поэтому total = excess."""
    spx = pd.read_parquet(spx_path).set_index("date").sort_index()
    close_col = "adj_close" if "adj_close" in spx.columns else "close"
    return spx[close_col].pct_change().rename("spx-rf")


def _build_presence_matrix(
    universe_path: str | Path,
    daily_index: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    """Расширить месячный universe в daily 0/1 матрицу на индексе `daily_index`.

    Правило: на день `d` тикер «активен», если он был в последнем
    universe-snapshot ≤ `d`. Реализация — pivot universe в month-end matrix,
    reindex на `daily_index` и `ffill`.
    """
    uni = pd.read_parquet(universe_path)
    uni["ticker"] = uni["ticker"].astype(str)
    uni["alive"] = 1
    me_matrix = uni.pivot_table(
        index="date", columns="ticker", values="alive",
        aggfunc="max", observed=True,
    ).fillna(0).astype("int8")
    me_matrix = me_matrix.reindex(columns=tickers, fill_value=0)
    presence = me_matrix.reindex(daily_index, method="ffill").fillna(0).astype("int8")
    return presence


def build_kaggle_dataset(
    config: KaggleUSExperimentConfig,
    prices_path: str | Path = "data/processed/prices.parquet",
    universe_path: str | Path = "data/processed/universe.parquet",
    spx_path: str | Path = "data/processed/spx.parquet",
) -> DatasetData:
    """Собрать `DatasetData` для подачи в `quant_pml.runner.build_backtest`.

    Что делаем:
      1. Pivot prices → wide `[date × ticker]` (close prices).
      2. Truncate на [DATA_PROCESSING_START_DATE - 1y, END_DATE], чтобы строить
         фичи и rolling windows. Минус 1 год — buffer для warm-up.
      3. Восстанавливаем business-day index (внутренние NaN — ffill, чтобы
         pct_change/factors не падали; presence_matrix решает, что торговать).
      4. Добавляем колонки `rf=0` и `spx-rf` = simple SPX return (RF=0).
      5. Ограничиваем universe тикерами, которые когда-либо появлялись в нашем
         monthly universe (это ~3000 имён вместо 6000+).
      6. Строим daily presence_matrix.
    """
    end_date = pd.Timestamp(config.END_DATE)
    buffer_start = pd.Timestamp(config.DATA_PROCESSING_START_DATE) - pd.DateOffset(years=1)

    logger.info("Loading prices wide...")
    prices_wide = _load_prices_wide(prices_path)
    prices_wide = prices_wide.loc[buffer_start:end_date]

    logger.info("Restricting universe to tickers seen in monthly universe...")
    uni = pd.read_parquet(universe_path, columns=["ticker"])
    uni_tickers = sorted(uni["ticker"].astype(str).unique().tolist())
    keep_tickers = [t for t in uni_tickers if t in prices_wide.columns]
    prices_wide = prices_wide.loc[:, keep_tickers]

    full_index = pd.bdate_range(prices_wide.index.min(), prices_wide.index.max())
    prices_wide = prices_wide.reindex(full_index).astype("float64")

    logger.info("Loading SPX and building factor column...")
    spx_excess = _load_spx_excess(spx_path).reindex(full_index)
    # ffill пропуски при несовпадении календарей (NYSE-holiday vs bdate_range);
    # остаточные NaN до начала SPX-серии заполняем нулём, чтобы OLS-бенчмарк
    # в Assessor не падал на MissingDataError.
    spx_excess = spx_excess.ffill().fillna(0.0)

    rf = pd.Series(0.0, index=full_index, name=config.RF_NAME)

    data = pd.concat([prices_wide, rf.to_frame(), spx_excess.to_frame()], axis=1)
    data.columns = data.columns.astype(str)
    data.index.name = "date"

    logger.info("Building presence_matrix (daily 0/1)...")
    presence = _build_presence_matrix(universe_path, full_index, keep_tickers)

    logger.info(
        "Dataset built: data=%s, presence=%s, tickers=%d, dates=%s..%s",
        data.shape, presence.shape, len(keep_tickers),
        full_index[0].date(), full_index[-1].date(),
    )

    return DatasetData(
        data=data,
        presence_matrix=presence,
        mkt_caps=None,
        dividends=None,
        volumes=None,
        targets=None,
        macro_features=None,
        asset_features=None,
    )
