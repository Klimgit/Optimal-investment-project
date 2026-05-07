"""Графики для оценки стратегий: equity, drawdown, rolling Sharpe, comparison.

Все функции возвращают `matplotlib.figure.Figure` — это удобно для:
- логирования в MLflow (`mlflow.log_figure(fig, "equity.png")`),
- сохранения в файл (`fig.savefig(path)`),
- inline-показа в notebook.

Стиль — sober quantitative: тонкая сетка, серая benchmark-линия,
log-шкала для equity (как принято в финансах).
"""
from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

_BENCHMARK_COLOR = "#888"
_STRATEGY_COLOR = "#2a72b3"
_FIGSIZE = (10, 5)


def _to_series(returns: pd.Series | pd.DataFrame, name: str = "strategy") -> pd.Series:
    if isinstance(returns, pd.DataFrame):
        return returns.iloc[:, 0].rename(name)
    return returns.rename(name)


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    eq = _equity_curve(returns)
    running_max = eq.cummax()
    return eq / running_max - 1.0


def plot_equity(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    *,
    strategy_name: str = "strategy",
    benchmark_name: str = "benchmark",
    log_scale: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """Кумулятивная equity-кривая в log-масштабе."""
    s = _to_series(strategy_returns, strategy_name)
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    eq_s = _equity_curve(s)
    ax.plot(eq_s.index, eq_s.values, label=strategy_name, color=_STRATEGY_COLOR, lw=1.4)

    if benchmark_returns is not None:
        b = _to_series(benchmark_returns, benchmark_name).reindex(s.index).fillna(0)
        eq_b = _equity_curve(b)
        ax.plot(eq_b.index, eq_b.values, label=benchmark_name, color=_BENCHMARK_COLOR, lw=1.0, ls="--")

    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_ylabel("Equity (NAV, log)" if log_scale else "Equity (NAV)")
    ax.set_xlabel("")
    ax.legend(frameon=False, loc="best")
    ax.set_title(title or f"Equity curve — {strategy_name}")
    fig.tight_layout()
    return fig


def plot_drawdown(
    strategy_returns: pd.Series | pd.DataFrame,
    *,
    strategy_name: str = "strategy",
    title: str | None = None,
) -> plt.Figure:
    """Просадка относительно running max NAV."""
    s = _to_series(strategy_returns, strategy_name)
    dd = _drawdown(s)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#c0392b", lw=1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_ylabel("Drawdown")
    ax.set_title(title or f"Drawdown — {strategy_name}")
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(
    strategy_returns: pd.Series | pd.DataFrame,
    window_days: int = 252,
    *,
    strategy_name: str = "strategy",
    title: str | None = None,
) -> plt.Figure:
    """Скользящий annualized Sharpe (252d)."""
    s = _to_series(strategy_returns, strategy_name)
    roll_mean = s.rolling(window_days).mean()
    roll_std = s.rolling(window_days).std()
    roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(roll_sharpe.index, roll_sharpe.values, color=_STRATEGY_COLOR, lw=1.2)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_ylabel(f"Rolling {window_days}d Sharpe (ann.)")
    ax.set_title(title or f"Rolling Sharpe — {strategy_name}")
    fig.tight_layout()
    return fig


def plot_weights_distribution(
    rebal_weights: pd.DataFrame,
    *,
    strategy_name: str = "strategy",
    title: str | None = None,
) -> plt.Figure:
    """Long/Short/Net exposure по времени."""
    pos = rebal_weights.where(rebal_weights > 0, 0).sum(axis=1)
    neg = rebal_weights.where(rebal_weights < 0, 0).sum(axis=1)
    net = rebal_weights.sum(axis=1)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.fill_between(pos.index, 0, pos.values, color="#27ae60", alpha=0.4, label="long")
    ax.fill_between(neg.index, neg.values, 0, color="#c0392b", alpha=0.4, label="short")
    ax.plot(net.index, net.values, color="black", lw=1.0, label="net")
    ax.axhline(0, color="black", lw=0.4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, loc="best")
    ax.set_title(title or f"Exposures — {strategy_name}")
    fig.tight_layout()
    return fig


def plot_comparison(
    strategies: Mapping[str, pd.Series | pd.DataFrame],
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    *,
    benchmark_name: str = "benchmark",
    log_scale: bool = True,
    title: str = "Strategies comparison",
) -> plt.Figure:
    """Несколько equity-кривых на одном графике."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    cmap = plt.get_cmap("tab10")

    for i, (name, ret) in enumerate(strategies.items()):
        s = _to_series(ret, name)
        eq = _equity_curve(s)
        ax.plot(eq.index, eq.values, label=name, color=cmap(i % 10), lw=1.3)

    if benchmark_returns is not None:
        # Берём пересечение индексов как универсальный окно для бенчмарка.
        all_idx = None
        for ret in strategies.values():
            s = _to_series(ret).index
            all_idx = s if all_idx is None else all_idx.intersection(s)
        b = _to_series(benchmark_returns, benchmark_name).reindex(all_idx).fillna(0)
        eq_b = _equity_curve(b)
        ax.plot(eq_b.index, eq_b.values, label=benchmark_name, color=_BENCHMARK_COLOR, lw=1.0, ls="--")

    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_ylabel("Equity (NAV, log)" if log_scale else "Equity (NAV)")
    ax.legend(frameon=False, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig
