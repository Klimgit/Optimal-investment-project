"""Holdout-OOS: срез дат для отчётных метрик без повторного бэктеста.

Рекомендуемый протокол (см. ``docs/holdout_protocol.md``):
гиперпараметры подбирать только на данных до ``HYPER_TUNING_END``;
финальный отчёт — на окне ``DEFAULT_EVAL_START`` … конец выборки.

Метрики здесь — **дневные** определения (Sharpe / IR / CAPM-α), чтобы их можно было
bootstrap-ить; они могут отличаться от ``metrics.csv`` из ``quant_pml`` Assessor
(там годовые величины из NAV за весь прогон).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

                                                                                        
HYPER_TUNING_END = pd.Timestamp("2015-12-31")
DEFAULT_EVAL_START = pd.Timestamp("2016-01-01")
DEFAULT_EVAL_END: pd.Timestamp | None = None                      


def slice_period(
    *series: pd.Series,
    start: pd.Timestamp | str | None,
    end: pd.Timestamp | str | None = None,
) -> tuple[pd.Series, ...]:
    """Общий календарный срез по индексу (после ``dropna`` конкатенации)."""
    ts_start = pd.Timestamp(start) if start is not None else None
    ts_end = pd.Timestamp(end) if end is not None else None
    out: list[pd.Series] = []
    for s in series:
        sx = s.copy()
        if ts_start is not None:
            sx = sx.loc[sx.index >= ts_start]
        if ts_end is not None:
            sx = sx.loc[sx.index <= ts_end]
        out.append(sx)
    return tuple(out)


def _align_two(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a.rename("s"), b.rename("b")], axis=1).dropna()
    return df["s"], df["b"]


def daily_sharpe_excess(strategy_total: pd.Series, rf: pd.Series | float = 0.0) -> float:
    """Sharpe по дневным доходностям: sqrt(252)*mean(xs)/std(xs), xs = total - rf."""
    rf_s = rf if isinstance(rf, pd.Series) else pd.Series(float(rf), index=strategy_total.index)
    xs = strategy_total.astype(float).sub(rf_s.reindex(strategy_total.index), fill_value=float(rf))
    xs = xs.dropna().to_numpy(dtype=float)
    return float(_ann_sharpe_from_daily(xs))


def _ann_sharpe_from_daily(xs: np.ndarray) -> float:
    xs = xs[np.isfinite(xs)]
    if xs.size < 2:
        return float("nan")
    mu, sig = float(np.mean(xs)), float(np.std(xs, ddof=1))
    if sig < 1e-12:
        return float("nan")
    return (mu / sig) * np.sqrt(252.0)


def daily_ir_vs_benchmark(
    strategy_total: pd.Series,
    benchmark_total: pd.Series,
    rf: pd.Series | float = 0.0,
) -> float:
    """IR по активной доходности: sqrt(252)*mean(s-b)/std(s-b), s,b — total vs rf как excess."""
    rf_s = rf if isinstance(rf, pd.Series) else pd.Series(float(rf), index=strategy_total.index)
    s, b = _align_two(
        strategy_total.astype(float).sub(rf_s.reindex(strategy_total.index), fill_value=float(rf)),
        benchmark_total.astype(float).sub(rf_s.reindex(benchmark_total.index), fill_value=float(rf)),
    )
    active = (s - b).to_numpy(dtype=float)
    active = active[np.isfinite(active)]
    if active.size < 2:
        return float("nan")
    mu, sig = float(np.mean(active)), float(np.std(active, ddof=1))
    if sig < 1e-12:
        return float("nan")
    return (mu / sig) * np.sqrt(252.0)


def daily_capm_alpha_annualized(
    strategy_total: pd.Series,
    benchmark_total: pd.Series,
    rf: pd.Series | float = 0.0,
) -> float:
    """CAPM на дневных доходностях: r_s - rf = α + β (r_b - rf); α annualized ≈ α_daily * 252."""
    rf_s = rf if isinstance(rf, pd.Series) else pd.Series(float(rf), index=strategy_total.index)
    s, b = _align_two(
        strategy_total.astype(float).sub(rf_s.reindex(strategy_total.index), fill_value=float(rf)),
        benchmark_total.astype(float).sub(rf_s.reindex(benchmark_total.index), fill_value=float(rf)),
    )
    y = s.to_numpy(dtype=float)
    x = b.to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    if y.size < 3:
        return float("nan")
    x_mean, y_mean = x.mean(), y.mean()
    cov = np.dot(x - x_mean, y - y_mean)
    var = np.dot(x - x_mean, x - x_mean)
    if var < 1e-18:
        return float("nan")
    beta = cov / var
    alpha_d = float(y_mean - beta * x_mean)
    return alpha_d * 252.0


def paired_daily_metrics_numpy(strategy_total: np.ndarray, benchmark_total: np.ndarray, rf: float = 0.0) -> dict[str, float]:
    """Те же метрики, что ``collect_holdout_metrics``, для выровненных numpy-массивов total returns."""
    st = np.asarray(strategy_total, dtype=float)
    bt = np.asarray(benchmark_total, dtype=float)
    mask = np.isfinite(st) & np.isfinite(bt)
    st, bt = st[mask], bt[mask]
    if st.size < 3:
        return {"sharpe_ann_daily_xs": float("nan"), "ir_benchmark_daily_active": float("nan"), "alpha_capm_ann": float("nan")}
    xs_s = st - rf
    xs_b = bt - rf
    sharpe = _ann_sharpe_from_daily(xs_s)
    active = xs_s - xs_b
    ir = _ann_sharpe_from_daily(active) if np.std(active, ddof=1) >= 1e-12 else float("nan")
                           
    x_mean, y_mean = float(xs_b.mean()), float(xs_s.mean())
    cov = float(np.dot(xs_b - x_mean, xs_s - y_mean))
    var = float(np.dot(xs_b - x_mean, xs_b - x_mean))
    if var < 1e-18:
        alpha_ann = float("nan")
    else:
        beta = cov / var
        alpha_d = y_mean - beta * x_mean
        alpha_ann = alpha_d * 252.0
    return {
        "sharpe_ann_daily_xs": float(sharpe),
        "ir_benchmark_daily_active": float(ir),
        "alpha_capm_ann": float(alpha_ann),
    }


def collect_holdout_metrics(
    strategy_total: pd.Series,
    benchmark_total: pd.Series,
    *,
    eval_start: pd.Timestamp | str | None,
    eval_end: pd.Timestamp | str | None = None,
    rf: pd.Series | float = 0.0,
) -> dict[str, Any]:
    """Точечные метрики на срезе [eval_start, eval_end]."""
    st, bt = slice_period(strategy_total, benchmark_total, start=eval_start, end=eval_end)
    st, bt = _align_two(st, bt)
    return {
        "eval_start": str(st.index.min().date()) if len(st) else None,
        "eval_end": str(st.index.max().date()) if len(st) else None,
        "n_days": int(len(st)),
        "sharpe_ann_daily_xs": daily_sharpe_excess(st, rf),
        "ir_benchmark_daily_active": daily_ir_vs_benchmark(st, bt, rf),
        "alpha_capm_ann": daily_capm_alpha_annualized(st, bt, rf),
    }


def resolve_benchmark_path(run_dir: Path) -> Path | None:
    """Ищем ``benchmark_returns.parquet`` рядом с run или у ``*_s0`` для ``*_agg``."""
    direct = run_dir / "benchmark_returns.parquet"
    if direct.exists():
        return direct
    name = run_dir.name
    if name.endswith("_agg"):
        base = name.removesuffix("_agg")
        s0 = run_dir.parent / f"{base}_s0" / "benchmark_returns.parquet"
        if s0.exists():
            return s0
    return None


def write_holdout_json(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    filename: str = "metrics_holdout.json",
) -> Path:
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
