"""Полный OOS-прогон бейзлайнов: Ridge (regression) и LogReg+L2 (classification).

Запуск:

    PYTHONPATH=. python scripts/run_baselines.py
    PYTHONPATH=. python scripts/run_baselines.py --strategies ridge logreg
    PYTHONPATH=. python scripts/run_baselines.py --start 2010-01-01 --end 2017-11-09
    PYTHONPATH=. python scripts/run_baselines.py --no-mlflow

Результаты:
- `results/{strategy}/`     — артефакты + 4 графика на стратегию
- `mlruns/`                 — MLflow-эксперимент (если установлен)

Для UI: `mlflow ui --backend-store-uri ./mlruns`
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Callable

import pandas as pd

from quant_pml.config.trading_config import TradingConfig
from quant_pml.runner import build_backtest

from src.backtest.dataset_adapter import build_kaggle_dataset
from src.backtest.experiment_config import KaggleUSExperimentConfig
from src.backtest.strategy import MLScoringStrategy
from src.evaluation.artifacts import save_run_artifacts
from src.evaluation.tracker import ExperimentTracker
from src.models.base import BaseModel
from src.models.logreg import LogRegL2Model
from src.models.ridge import RidgeModel
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _strategy_specs() -> dict[str, dict]:
    """Реестр доступных стратегий: name → {model_factory, target_col, params}."""
    return {
        "ridge": {
            "model_factory": lambda: RidgeModel(alpha=10.0),
            "target_col": "target_reg",
            "model_params": {"model": "Ridge", "alpha": 10.0},
        },
        "logreg": {
            "model_factory": lambda: LogRegL2Model(C=1.0, max_iter=1000),
            "target_col": "target_clf",
            "model_params": {"model": "LogisticRegression", "C": 1.0, "max_iter": 1000},
        },
    }


def _run_single(
    strategy_name: str,
    model_factory: Callable[[], BaseModel],
    target_col: str,
    model_params: dict,
    cfg: KaggleUSExperimentConfig,
    trading_cfg: TradingConfig,
    tracker: ExperimentTracker,
    train_window_months: int = 60,
    quantile: float = 0.1,
) -> dict:
    """Прогнать одну стратегию end-to-end и сохранить артефакты."""
    logger.info("=" * 70)
    logger.info("Strategy: %s | target=%s | window=%d mo", strategy_name, target_col, train_window_months)
    logger.info("OOS: %s..%s | rebal=%s", cfg.START_DATE.date(), cfg.END_DATE.date(), cfg.REBALANCE_FREQ)
    logger.info("=" * 70)
    t0 = time.time()

    strategy = MLScoringStrategy(
        model_factory=model_factory,
        target_col=target_col,
        train_window_months=train_window_months,
        mode="long_short",
        quantile=quantile,
    )

    preprocessor, runner = build_backtest(
        experiment_config=cfg,
        trading_config=trading_cfg,
        rebal_freq=cfg.REBALANCE_FREQ,
        dataset_builder_fn=build_kaggle_dataset,
        verbose=True,
    )
    stats = runner.run(preprocessor, strategy, hedger=None)

    elapsed = time.time() - t0
    logger.info("Backtest finished in %.1fs", elapsed)

    paths = save_run_artifacts(strategy_name, runner, stats, output_dir="results")

    params = {
        "strategy": strategy_name,
        "target_col": target_col,
        "train_window_months": train_window_months,
        "quantile": quantile,
        "mode": "long_short",
        "rebal_freq": cfg.REBALANCE_FREQ,
        "start": str(cfg.START_DATE.date()),
        "end": str(cfg.END_DATE.date()),
        "broker_fee": trading_cfg.broker_fee,
        "trading_lag_days": trading_cfg.trading_lag_days,
        **model_params,
    }
    metrics = {
        "sharpe": stats.sharpe,
        "geom_avg_total_r": stats.geom_avg_total_r,
        "geom_avg_xs_r": stats.geom_avg_xs_r,
        "std_xs_r": stats.std_xs_r,
        "max_dd": stats.max_dd,
        "alpha_benchmark": stats.alpha_benchmark,
        "alpha_benchmark_pvalue": stats.alpha_benchmark_pvalue,
        "ir_benchmark": stats.ir_benchmark,
        "alpha_buy_hold": stats.alpha_buy_hold,
        "ir_buy_hold": stats.ir_buy_hold,
        "final_nav": stats.final_nav,
        "skew": stats.skew,
        "kurtosis": stats.kurtosis,
        "elapsed_sec": elapsed,
    }

    with tracker.run(strategy_name, params=params) as t:
        t.log_metrics(metrics)
        for img in ("equity", "drawdown", "rolling_sharpe", "exposures"):
            t.log_artifact(paths[img])
        t.log_artifact(paths["metrics_csv"])
        t.log_artifact(paths["metrics_json"])

    return {
        "strategy": strategy_name,
        "metrics": metrics,
        "params": params,
        "paths": paths,
    }


def _print_summary(results: list[dict]) -> None:
    if not results:
        return
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    cols = ["sharpe", "geom_avg_total_r", "max_dd", "alpha_benchmark", "ir_benchmark", "final_nav"]
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append([r["strategy"]] + [m[c] for c in cols])
    df = pd.DataFrame(rows, columns=["strategy", *cols])
    logger.info("\n%s", df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["ridge", "logreg"], choices=["ridge", "logreg"])
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2017-11-09")
    parser.add_argument("--train-window", type=int, default=60)
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--broker-fee", type=float, default=0.0005, help="bps как доля; 0.0005 = 5bp")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="dl-momentum-baselines")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    cfg = KaggleUSExperimentConfig()
    cfg.START_DATE = pd.Timestamp(args.start)
    cfg.END_DATE = pd.Timestamp(args.end)

    trading_cfg = TradingConfig(
        broker_fee=args.broker_fee,
        bid_ask_spread=0.0,
        total_exposure=1.0,
        max_exposure=None,
        min_exposure=None,
        trading_lag_days=1,
    )

    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        enabled=not args.no_mlflow,
    )

    specs = _strategy_specs()
    results = []
    for name in args.strategies:
        spec = specs[name]
        cfg_copy = KaggleUSExperimentConfig()
        cfg_copy.START_DATE = cfg.START_DATE
        cfg_copy.END_DATE = cfg.END_DATE
        try:
            res = _run_single(
                strategy_name=name,
                model_factory=spec["model_factory"],
                target_col=spec["target_col"],
                model_params=spec["model_params"],
                cfg=cfg_copy,
                trading_cfg=trading_cfg,
                tracker=tracker,
                train_window_months=args.train_window,
                quantile=args.quantile,
            )
            results.append(res)
        except Exception as e:
            logger.exception("Strategy %s failed: %s", name, e)

    _print_summary(results)


if __name__ == "__main__":
    main()
