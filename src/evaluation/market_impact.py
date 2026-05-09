"""Пост-хок модели market impact (движок quant-pml не моделирует impact явно).

Линейная эвристика по обороту (доля портфеля, перебалансируемая за день):

    adjusted_return_t ≈ raw_return_t - η · turnover_t

где ``turnover_t`` — сумма абсолютных изменений весов / масштаб по вашим данным.

Использование: построить ``turnover`` из ``rebal_weights.parquet`` или daily weights,
затем вычесть из сохранённых ``total_returns`` перед ``holdout_metrics`` / bootstrap.

Полная калибровка η и нелинейный impact — исследовательская задача (см. ``docs/advanced_ml_backlog.md``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def approximate_daily_turnover_fraction(
    rebal_weights: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.Series:
    """Серия дневного **оборота** (доля портфеля, 0..~2 на сторону).

    На каждом ребалансе: ``0.5 * sum_i |w_i - w_i_prev|``; оборот **равномерно
    размазан** по торговым дням от даты ребаланса (вкл.) до следующего ребаланса
    (не вкл.). Первый ребаланс сравнивается с нулевыми весами.
    """
    r = rebal_weights.sort_index()
    if r.empty:
        return pd.Series(0.0, index=daily_index)
    dix = daily_index.sort_values()
    dmin, dmax = dix.min(), dix.max()
    dates = r.index
    w = r.to_numpy(dtype=float)
    n = len(dates)
    events: list[tuple[pd.Timestamp, float]] = []
    for i in range(n):
        if i == 0:
            to = 0.5 * float(np.abs(w[0]).sum())
        else:
            to = 0.5 * float(np.abs(w[i] - w[i - 1]).sum())
        d = dates[i]
        if d < dmin or d > dmax:
            continue
        events.append((pd.Timestamp(d), to))

    out = pd.Series(0.0, index=dix, dtype=float)
    for i, (d0, to) in enumerate(events):
        d1 = events[i + 1][0] if i + 1 < len(events) else dmax + pd.Timedelta(days=1)
        m = (out.index >= d0) & (out.index < d1)
        sub = out.index[m]
        if len(sub) == 0:
            continue
        out.loc[sub] = to / float(len(sub))
    return out.reindex(daily_index).fillna(0.0)


def linear_impact_haircut(
    daily_returns: pd.Series,
    turnover: pd.Series,
    *,
    eta: float,
) -> pd.Series:
    """Вычитание η·turnover (выравниваем по индексу, пропуски в turnover → 0)."""
    t = turnover.reindex(daily_returns.index).fillna(0.0).astype(float)
    return daily_returns.astype(float).sub(t * eta)
