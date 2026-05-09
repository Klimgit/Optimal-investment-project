"""Сохранение артефактов одного backtest-прогона.

После `runner.run(...)` у нас в руках:
- `runner.strategy_total_r`     — daily total returns (DataFrame, col `total_r`)
- `runner.strategy_excess_r`    — daily excess returns (DataFrame, col `excess_r`)
- `runner.strategy_rebal_weights`  — веса на ребаланс-датах
- `runner.strategy_daily_weights`  — daily forward-filled веса
- `stats: StrategyStatistics`   — итоговые метрики

Эта функция складывает всё в единую папку `results/{strategy_name}/` и
дополнительно строит четыре сводных графика. Возвращает dict с путями —
чтобы вызывающий мог при желании всё это залить в MLflow.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quant_pml.backtest.assessor import StrategyStatistics

from src.evaluation.plots import (
    plot_drawdown,
    plot_equity,
    plot_rolling_sharpe,
    plot_weights_distribution,
)

logger = logging.getLogger(__name__)


def _benchmark_total_returns(runner) -> pd.Series:
    """SPX total return = factor (excess) + RF. У нас RF=0, так что это spx-rf."""
    factor_name = runner.experiment_config.MKT_NAME
    factors = runner.factors[factor_name]
    rf = runner.rf
    common_idx = factors.index.intersection(rf.index)
    return (factors.loc[common_idx] + rf.loc[common_idx]).rename("benchmark")


def _stats_to_dict(stats: StrategyStatistics) -> dict[str, Any]:
    """Распаковать dataclass + раскрыть `factor_loadings` в плоские поля."""
    out: dict[str, Any] = {}
    for k, v in asdict(stats).items():
        if k == "factor_loadings" and isinstance(v, dict):
            for fk, fv in v.items():
                out[f"loading_{fk}"] = fv
        elif k == "strategy_name":
            continue
        else:
            out[k] = v
    return out


def save_run_artifacts(
    strategy_name: str,
    runner,
    stats: StrategyStatistics,
    output_dir: str | Path = "results",
    *,
    save_daily_weights: bool = False,
) -> dict[str, Path]:
    """Сохранить артефакты прогона в `{output_dir}/{strategy_name}/`.

    Структура папки:

        results/{strategy_name}/
            metrics.json
            metrics.csv
            total_returns.parquet
            excess_returns.parquet
            rebal_weights.parquet
            benchmark_returns.parquet
            equity.png
            drawdown.png
            rolling_sharpe.png
            exposures.png
    """
    out = Path(output_dir) / strategy_name
    out.mkdir(parents=True, exist_ok=True)

    metrics = _stats_to_dict(stats)
    with (out / "metrics.json").open("w") as f:
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float, str, type(None)))}, f, indent=2)
    pd.Series(metrics, name="value").to_csv(out / "metrics.csv", header=True)

    total_r = runner.strategy_total_r.copy()
    excess_r = runner.strategy_excess_r.copy()
    rebal_w = runner.strategy_rebal_weights.copy()
    bench = _benchmark_total_returns(runner).to_frame()

    total_r.to_parquet(out / "total_returns.parquet")
    excess_r.to_parquet(out / "excess_returns.parquet")
    rebal_w.to_parquet(out / "rebal_weights.parquet")
    bench.to_parquet(out / "benchmark_returns.parquet")

    if save_daily_weights:
        runner.strategy_daily_weights.to_parquet(out / "daily_weights.parquet")

    bench_aligned = bench.iloc[:, 0].reindex(total_r.index).fillna(0)

    fig_eq = plot_equity(total_r, bench_aligned, strategy_name=strategy_name, benchmark_name="SPX (total)")
    fig_eq.savefig(out / "equity.png", dpi=120)

    fig_dd = plot_drawdown(total_r, strategy_name=strategy_name)
    fig_dd.savefig(out / "drawdown.png", dpi=120)

    fig_rs = plot_rolling_sharpe(total_r, window_days=252, strategy_name=strategy_name)
    fig_rs.savefig(out / "rolling_sharpe.png", dpi=120)

    fig_exp = plot_weights_distribution(rebal_w, strategy_name=strategy_name)
    fig_exp.savefig(out / "exposures.png", dpi=120)

    import matplotlib.pyplot as plt
    for fig in (fig_eq, fig_dd, fig_rs, fig_exp):
        plt.close(fig)

    paths = {
        "metrics_json": out / "metrics.json",
        "metrics_csv": out / "metrics.csv",
        "total_returns": out / "total_returns.parquet",
        "excess_returns": out / "excess_returns.parquet",
        "rebal_weights": out / "rebal_weights.parquet",
        "benchmark_returns": out / "benchmark_returns.parquet",
        "equity": out / "equity.png",
        "drawdown": out / "drawdown.png",
        "rolling_sharpe": out / "rolling_sharpe.png",
        "exposures": out / "exposures.png",
    }
    logger.info("Saved %d artifacts to %s", len(paths), out)
    return paths
