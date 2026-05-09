"""Circular block bootstrap для дневных доходностей (устойчивость к автокорреляции).

Одиночный ряд: Sharpe по strategy total.

Парный ряд (strategy, benchmark): совместный блоковый ресэмплинг строк —
Sharpe(xs), IR по активной доходности, CAPM-α (annualized), см. ``holdout.paired_daily_metrics_numpy``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.evaluation.holdout import paired_daily_metrics_numpy


def annualized_sharpe_daily(daily_r: np.ndarray) -> float:
    x = np.asarray(daily_r, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=1))
    if sig < 1e-12:
        return float("nan")
    return (mu / sig) * np.sqrt(252.0)


def block_bootstrap_metric_ci(
    daily_returns: pd.Series,
    *,
    metric: Literal["sharpe"] = "sharpe",
    n_bootstrap: int = 2000,
    block_len: int = 21,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> dict[str, float]:
    """Circular block bootstrap; возвращает точечную оценку и квантили bootstrap-распределения."""
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    arr = r.to_numpy(dtype=float)
    n = arr.size
    if n < 2:
        raise ValueError("need at least 2 daily returns")

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(
        random_state
    )
    bl = max(1, min(block_len, n))

    def one_sample() -> np.ndarray:
        out: list[float] = []
        while len(out) < n:
            start = int(rng.integers(0, n))
            for j in range(bl):
                out.append(arr[(start + j) % n])
        return np.asarray(out[:n], dtype=float)

    if metric != "sharpe":
        raise ValueError(f"unsupported metric: {metric}")

    point = annualized_sharpe_daily(arr)
    stats_boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        stats_boot[i] = annualized_sharpe_daily(one_sample())

    q_lo, q_hi = np.quantile(stats_boot, [alpha / 2, 1.0 - alpha / 2])
    return {
        "point": float(point),
        "ci_low": float(q_lo),
        "ci_high": float(q_hi),
        "n_days": float(n),
        "block_len": float(bl),
        "n_bootstrap": float(n_bootstrap),
        "alpha": float(alpha),
    }


def block_bootstrap_paired_ci(
    strategy_total: pd.Series,
    benchmark_total: pd.Series,
    *,
    rf: float = 0.0,
    n_bootstrap: int = 2000,
    block_len: int = 21,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> dict[str, dict[str, float]]:
    """Парный circular block bootstrap по выровненным (s,b)."""
    df = pd.concat(
        [strategy_total.rename("s"), benchmark_total.rename("b")],
        axis=1,
    ).dropna()
    if len(df) < 3:
        raise ValueError("need aligned strategy/benchmark with >= 3 rows")
    mat = df.to_numpy(dtype=float)
    n, _ = mat.shape

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(
        random_state
    )
    bl = max(1, min(block_len, n))

    def one_sample() -> np.ndarray:
        out: list[list[float]] = []
        while len(out) < n:
            start = int(rng.integers(0, n))
            for j in range(bl):
                row = mat[(start + j) % n]
                out.append([float(row[0]), float(row[1])])
        return np.asarray(out[:n], dtype=float)

    keys = ("sharpe_ann_daily_xs", "ir_benchmark_daily_active", "alpha_capm_ann")
    point = paired_daily_metrics_numpy(mat[:, 0], mat[:, 1], rf=rf)

    boot = {k: np.empty(n_bootstrap, dtype=float) for k in keys}
    for i in range(n_bootstrap):
        samp = one_sample()
        m = paired_daily_metrics_numpy(samp[:, 0], samp[:, 1], rf=rf)
        for k in keys:
            boot[k][i] = m[k]

    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        ql, qh = np.quantile(boot[k], [q_lo, q_hi])
        out[k] = {
            "point": float(point[k]),
            "ci_low": float(ql),
            "ci_high": float(qh),
            "n_days": float(n),
            "block_len": float(bl),
            "n_bootstrap": float(n_bootstrap),
            "alpha": float(alpha),
        }
    return out
