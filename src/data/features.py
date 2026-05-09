"""Построение признаков и таргетов для DL-momentum-стратегии.

24 признака на актив (+ опционально 5 regime по SPX из ``paths.benchmark``):
- 8 momentum:  r1, r3, r6, r12, sigma_ann, r3_n, r6_n, r12_n
- 16 MACD/Signal: для 8 пар EMA (fast, slow):
    macd_{fast}_{slow} = (EMA_fast - EMA_slow) / price
    sig_{fast}_{slow}  = EMA_9(macd) / 1   (Signal линия)
Regime: ``reg_spx_ret21/63``, ``reg_spx_vol21/63``, ``reg_spx_dd126`` —
без cross-section Z-score (одинаковы по всем тикерам на дату снимка).

Таргеты на месячный snapshot d:
    ret_next   = close[next_d] / close[d] - 1
    target_reg = ret_next - mean(ret_next over universe at d)   (ex-mean)
    target_clf = 1 если ret_next в верхнем дециле, 0 если в нижнем, NaN иначе

Cross-sectional Z-score применяется **внутри каждой ребаланс-даты** (только
поперечное сечение на эту дату — без будущих календарных месяцев). Это согласуется
с требованием «нормировать без утечки OOS по времени» при пошаговом walk-forward:
на каждом шаге скоринг использует фичи, известные на дату снимка. Помесячный
retrain на скользящем окне ``train_window_months`` реализован в
``MLScoringStrategy`` / ``quant_pml`` (как на слайдах методички).

Запуск как модуль:

    python -m src.data.features [--config base.yaml] [--out data/features/panel.parquet]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.io import load_config, read_parquet, write_parquet

logger = logging.getLogger(__name__)


                                       

MOMENTUM_COLS: list[str] = ["r1", "r3", "r6", "r12", "sigma_ann", "r3_n", "r6_n", "r12_n"]


def macd_pair_names(macd_pairs: Iterable[tuple[int, int]]) -> tuple[list[str], list[str]]:
    macd_cols = [f"macd_{f}_{s}" for f, s in macd_pairs]
    sig_cols = [f"sig_{f}_{s}" for f, s in macd_pairs]
    return macd_cols, sig_cols


def feature_columns(macd_pairs: Iterable[tuple[int, int]]) -> list[str]:
    """Все 24 признака в каноническом порядке."""
    macd, sig = macd_pair_names(macd_pairs)
    return list(MOMENTUM_COLS) + macd + sig


                                                                                  

REGIME_FEATURE_COLS: list[str] = [
    "reg_spx_ret21",
    "reg_spx_ret63",
    "reg_spx_vol21",
    "reg_spx_vol63",
    "reg_spx_dd126",
]


def compute_spx_regime_series(spx_close: pd.Series) -> pd.DataFrame:
    """Дневные regime-признаки по SPX, строго causal (только прошлое и текущий бар).

    Столбцы: доходности SPX, аннуализованная вола дневных ретёрнов, просадка от
    максимума за 126 торговых дней. Индекс совпадает с ``spx_close.index``.
    """
    px = spx_close.sort_index().astype(float)
    r1 = px.pct_change()
    ret21 = px.pct_change(21)
    ret63 = px.pct_change(63)
    vol21 = r1.rolling(21, min_periods=10).std() * np.sqrt(252.0)
    vol63 = r1.rolling(63, min_periods=21).std() * np.sqrt(252.0)
    roll_max = px.rolling(126, min_periods=63).max()
    dd126 = px / roll_max.replace(0.0, np.nan) - 1.0
    return pd.DataFrame(
        {
            "reg_spx_ret21": ret21,
            "reg_spx_ret63": ret63,
            "reg_spx_vol21": vol21,
            "reg_spx_vol63": vol63,
            "reg_spx_dd126": dd126,
        },
        index=px.index,
    )


def regime_row_asof(regime_daily: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    """Последняя строка regime на дату ``d`` включительно (нет заглядывания вперёд)."""
    ts = pd.Timestamp(d)
    sl = regime_daily.loc[:ts]
    if sl.empty:
        return pd.Series({c: np.nan for c in regime_daily.columns})
    return sl.iloc[-1]


                                                        

def compute_daily_features(
    prices: pd.DataFrame,
    macd_pairs: Iterable[tuple[int, int]],
    signal_ema_days: int = 9,
    vol_lookback_days: int = 63,
    vol_annualization: int = 252,
) -> pd.DataFrame:
    """Добавить 24 признака к копии `prices`.

    `prices` ожидается в long-формате: `[date, ticker, open, high, low, close, volume]`.
    EMA / rolling / pct_change считаются по группам `ticker`, чтобы не «протекало»
    между активами. NaN в первых строках каждого тикера — норма (rolling warm-up).
    """
    df = prices.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    g_close = df.groupby("ticker", observed=True)["close"]

    df["ret_1d"] = g_close.pct_change(1)

    df["r1"] = g_close.pct_change(21)
    df["r3"] = g_close.pct_change(63)
    df["r6"] = g_close.pct_change(126)
    df["r12"] = g_close.pct_change(252)

    df["sigma_ann"] = (
        df.groupby("ticker", observed=True)["ret_1d"]
        .transform(lambda s: s.rolling(vol_lookback_days, min_periods=vol_lookback_days).std())
        * np.sqrt(vol_annualization)
    )

    sigma_safe = df["sigma_ann"].replace(0.0, np.nan)
    df["r3_n"] = df["r3"] / sigma_safe
    df["r6_n"] = df["r6"] / sigma_safe
    df["r12_n"] = df["r12"] / sigma_safe

    close_safe = df["close"].replace(0.0, np.nan)
    pairs = list(macd_pairs)
    for fast, slow in pairs:
        ema_fast = g_close.transform(
            lambda s, span=fast: s.ewm(span=span, adjust=False).mean()
        )
        ema_slow = g_close.transform(
            lambda s, span=slow: s.ewm(span=span, adjust=False).mean()
        )
        macd = (ema_fast - ema_slow) / close_safe
        df[f"macd_{fast}_{slow}"] = macd
        df[f"sig_{fast}_{slow}"] = (
            macd.groupby(df["ticker"], observed=True)
            .transform(lambda s, span=signal_ema_days: s.ewm(span=span, adjust=False).mean())
        )

    df = df.drop(columns=["ret_1d"])
    return df


                                              

def build_monthly_panel(
    daily: pd.DataFrame,
    universe: pd.DataFrame,
    feature_cols: list[str],
    top_q: float = 0.1,
    bottom_q: float = 0.1,
    regime_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Срезать daily на ребаланс-датах из universe и добавить таргеты.

    Возвращает DataFrame `[date, ticker, <feature_cols>, ret_next, target_reg, target_clf]`.
    """
    daily = daily.copy()
    daily["ticker"] = daily["ticker"].astype(str)
    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str)

    rebal_dates = pd.DatetimeIndex(sorted(universe["date"].unique()))

    uni_by_date: dict[pd.Timestamp, set[str]] = {
        d: set(g["ticker"].tolist()) for d, g in universe.groupby("date", observed=True)
    }

    close_wide = daily.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last", observed=True,
    )

    daily_idx = daily.set_index(["date", "ticker"]).sort_index()

    parts: list[pd.DataFrame] = []
    for i, d in enumerate(rebal_dates):
        tix = uni_by_date.get(d, set())
        if not tix or d not in close_wide.index:
            continue

        try:
            snap_full = daily_idx.loc[d]
        except KeyError:
            continue
        if isinstance(snap_full, pd.Series):
            snap_full = snap_full.to_frame().T

        snap_full = snap_full.loc[snap_full.index.isin(tix)]
        if snap_full.empty:
            continue

        snap = snap_full[feature_cols].copy()
        snap.insert(0, "ticker", snap.index.astype(str))
        snap.insert(0, "date", d)
        snap = snap.reset_index(drop=True)

        if regime_daily is not None:
            reg = regime_row_asof(regime_daily, pd.Timestamp(d))
            for col in REGIME_FEATURE_COLS:
                if col in regime_daily.columns:
                    v = reg.get(col, np.nan)
                    snap[col] = float(v) if pd.notna(v) else np.nan

        if i + 1 < len(rebal_dates):
            next_d = rebal_dates[i + 1]
            if next_d in close_wide.index:
                close_d = close_wide.loc[d]
                close_next = close_wide.loc[next_d]
                ret = (close_next / close_d - 1.0).reindex(snap["ticker"]).to_numpy()
                snap["ret_next"] = ret
            else:
                snap["ret_next"] = np.nan
        else:
            snap["ret_next"] = np.nan

        if snap["ret_next"].notna().sum() > 0:
            mean_ret = snap["ret_next"].mean(skipna=True)
            snap["target_reg"] = snap["ret_next"] - mean_ret
            top_thr = snap["ret_next"].quantile(1 - top_q)
            bot_thr = snap["ret_next"].quantile(bottom_q)
            snap["target_clf"] = np.where(
                snap["ret_next"] >= top_thr, 1.0,
                np.where(snap["ret_next"] <= bot_thr, 0.0, np.nan),
            )
        else:
            snap["target_reg"] = np.nan
            snap["target_clf"] = np.nan

        snap = snap.dropna(subset=feature_cols, how="all")
        parts.append(snap)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


                                               

def zscore_within_date(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Z-score по столбцам `feature_cols` внутри каждой `date`-группы."""
    out = panel.copy()
    for col in feature_cols:
        g = out.groupby("date", observed=True)[col]
        mu = g.transform("mean")
        sigma = g.transform("std").replace(0.0, np.nan)
        out[col] = (out[col] - mu) / sigma
    return out


                                  

def build_features(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Полный пайплайн: `prices` + `universe` → нормированный monthly panel."""
    macd_pairs = [tuple(p) for p in cfg["features"]["macd_pairs"]]
    feat_cols = feature_columns(macd_pairs)

    regime_daily: pd.DataFrame | None = None
    paths = cfg.get("paths") or {}
    bench = paths.get("benchmark")
    if bench and Path(str(bench)).exists():
        spx_tab = read_parquet(bench)
        if "date" in spx_tab.columns:
            spx_tab = spx_tab.set_index("date")
        spx_tab = spx_tab.sort_index()
        close_col = "adj_close" if "adj_close" in spx_tab.columns else "close"
        regime_daily = compute_spx_regime_series(spx_tab[close_col].astype(float))
        logger.info("Regime features from %s (%d rows)", bench, len(regime_daily))
    else:
        logger.info("Benchmark path missing or file not found — regime features skipped")

    logger.info("Computing daily features (%d MACD pairs, vol=%dd)...",
                len(macd_pairs), cfg["features"]["vol_lookback_days"])
    daily = compute_daily_features(
        prices,
        macd_pairs=macd_pairs,
        signal_ema_days=cfg["features"]["signal_ema_days"],
        vol_lookback_days=cfg["features"]["vol_lookback_days"],
        vol_annualization=cfg["features"]["vol_annualization"],
    )

    logger.info("Building monthly panel from %d universe rebalance dates...",
                universe["date"].nunique())
    panel = build_monthly_panel(
        daily,
        universe,
        feature_cols=feat_cols,
        top_q=cfg["target"]["classification"]["top_q"],
        bottom_q=cfg["target"]["classification"]["bottom_q"],
        regime_daily=regime_daily,
    )

    logger.info(
        "Z-scoring equity features within each rebalance date (%d cols; regime cols raw)...",
        len(feat_cols),
    )
    panel = zscore_within_date(panel, feat_cols)
    return panel


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build features panel from prices and universe")
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--universe", default="data/processed/universe.parquet")
    ap.add_argument("--out", default="data/features/panel.parquet")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prices_path = cfg["paths"]["processed_prices"]
    if not Path(prices_path).exists():
        raise FileNotFoundError(f"{prices_path} не найден; запустите `python -m src.data.loader`")
    if not Path(args.universe).exists():
        raise FileNotFoundError(
            f"{args.universe} не найден; запустите `python -m src.data.universe`"
        )

    prices = read_parquet(prices_path)
    universe = read_parquet(args.universe)

    panel = build_features(prices, universe, cfg)
    write_parquet(panel, args.out)
    logger.info("Wrote %d rows × %d cols to %s", len(panel), panel.shape[1], args.out)


if __name__ == "__main__":
    main()
