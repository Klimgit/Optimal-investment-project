"""Детерминированные факторные бенчмарки в нашем universe (low-vol L/S).

Академические линии SMB/HML/RMW (Fama–French) подмешиваются на график в
`scripts/compare_runs.py` из кеша; этот скрипт добавляет **low-risk в нашем
универсе**: long низкая σ, short высокая σ (после кросс-секционного Z-score
в панели — скор `−sigma_ann`).

Запуск:

    PYTHONPATH=. python scripts/run_factor_benchmarks.py
    PYTHONPATH=. python scripts/run_factor_benchmarks.py --start 2010-01-01 --end 2017-11-09
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from quant_pml.config.trading_config import TradingConfig
from quant_pml.runner import build_backtest

from src.backtest.dataset_adapter import build_kaggle_dataset
from src.backtest.experiment_config import KaggleUSExperimentConfig
from src.backtest.factor_column_strategy import FactorColumnStrategy
from src.evaluation.artifacts import save_run_artifacts

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2010-01-01")
    ap.add_argument("--end", type=str, default="2017-11-09")
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--broker-fee", type=float, default=0.0005)
    ap.add_argument("--panel", type=str, default="data/features/panel.parquet")
    ap.add_argument("--strategies", nargs="+", default=["bench_low_vol_ls"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    specs = {
        "bench_low_vol_ls": FactorColumnStrategy(
            "sigma_ann",
            panel_path=args.panel,
            score_sign=-1.0,
            mode="long_short",
            quantile=args.quantile,
        ),
    }

    cfg = KaggleUSExperimentConfig()
    cfg.START_DATE = pd.Timestamp(args.start)
    cfg.END_DATE = pd.Timestamp(args.end)

    trading_cfg = TradingConfig(
        broker_fee=args.broker_fee,
        bid_ask_spread=0.0,
        total_exposure=1.0,
        trading_lag_days=1,
    )

    for name in args.strategies:
        if name not in specs:
            raise ValueError(f"Unknown strategy {name}; choose from {list(specs)}")
        strategy = specs[name]
        logger.info("Running factor benchmark: %s", name)
        t0 = time.time()
        preprocessor, runner = build_backtest(
            experiment_config=cfg,
            trading_config=trading_cfg,
            rebal_freq=cfg.REBALANCE_FREQ,
            dataset_builder_fn=build_kaggle_dataset,
            verbose=True,
        )
        stats = runner.run(preprocessor, strategy, hedger=None)
        save_run_artifacts(name, runner, stats, output_dir="results")
        logger.info("%s finished in %.1fs (Sharpe=%.4f NAV=%.4f)", name, time.time() - t0, stats.sharpe, stats.final_nav)


if __name__ == "__main__":
    main()
