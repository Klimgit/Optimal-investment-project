"""Smoke-test полного pipeline Фазы 3 на 12 месяцах OOS с Ridge.

Цель — убедиться, что:
- наш `dataset_adapter` собирает корректный `DatasetData`,
- `MLScoringStrategy` обучается на panel и выдаёт скоры,
- `quant_pml.runner.build_backtest` успешно прогоняет backtest.

После успеха перейдём к Фазе 4 (Ridge/LogReg на полном OOS).

Запуск:

    python scripts/smoke_backtest.py
"""
from __future__ import annotations

import logging

import pandas as pd

from quant_pml.config.trading_config import TradingConfig
from quant_pml.runner import build_backtest

from src.backtest.dataset_adapter import build_kaggle_dataset
from src.backtest.experiment_config import KaggleUSExperimentConfig
from src.backtest.strategy import MLScoringStrategy
from src.models.ridge import RidgeModel
from src.utils.seed import set_seed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(0)

    cfg = KaggleUSExperimentConfig()
    cfg.START_DATE = pd.Timestamp("2010-01-01")
    cfg.END_DATE = pd.Timestamp("2011-01-31")

    trading_cfg = TradingConfig(
        broker_fee=0.0005,
        bid_ask_spread=0.0,
        total_exposure=1.0,
        max_exposure=None,
        min_exposure=None,
        trading_lag_days=1,
    )

    strategy = MLScoringStrategy(
        model_factory=lambda: RidgeModel(alpha=1.0),
        target_col="target_reg",
        train_window_months=60,
        mode="long_short",
        quantile=0.1,
    )

    preprocessor, runner = build_backtest(
        experiment_config=cfg,
        trading_config=trading_cfg,
        rebal_freq="ME",
        dataset_builder_fn=build_kaggle_dataset,
        verbose=True,
    )

    stats = runner.run(preprocessor, strategy, hedger=None)

    print("\n=== Smoke backtest done ===")
    print(f"Sharpe:           {stats.sharpe:.3f}")
    print(f"Geom mean total:  {stats.geom_avg_total_r:.4%}  (annualized)")
    print(f"Geom mean excess: {stats.geom_avg_xs_r:.4%}")
    print(f"Vol (excess):     {stats.std_xs_r:.4%}")
    print(f"Max drawdown:     {stats.max_dd:.4%}")
    print(f"Alpha vs benchmark: {stats.alpha_benchmark:.4%} (p={stats.alpha_benchmark_pvalue:.3f})")
    print(f"IR vs benchmark:    {stats.ir_benchmark:.3f}")
    print(f"Final NAV:        {stats.final_nav:.4f}")
    print(f"Total returns rows: {len(runner.strategy_total_r)}")
    print(f"Rebalances:       {len(runner.strategy_rebal_weights)}")


if __name__ == "__main__":
    main()
