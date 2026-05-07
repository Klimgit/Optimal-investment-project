"""Прогон DL-регрессоров (MLP, LSTM) на полном OOS с N сидами.

Каждый сид — отдельный backtest-run (полный retrain каждого ребаланса со
своим сидом). После всех сидов считаем агрегаты: mean returns, ±std band,
average metrics. Это честнее, чем усреднение скоров внутри одного бэктеста.

Запуск:

    PYTHONPATH=. python scripts/run_dl_reg.py --strategies mlp lstm --seeds 5
    PYTHONPATH=. python scripts/run_dl_reg.py --strategies mlp --seeds 3 --start 2014-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_pml.config.trading_config import TradingConfig
from quant_pml.runner import build_backtest

from src.backtest.dataset_adapter import build_kaggle_dataset
from src.backtest.experiment_config import KaggleUSExperimentConfig
from src.backtest.strategy import MLScoringStrategy
from src.evaluation.artifacts import save_run_artifacts
from src.evaluation.plots import plot_comparison, plot_drawdown, plot_equity, plot_rolling_sharpe
from src.evaluation.tracker import ExperimentTracker
from src.models.base import BaseModel
from src.models.lstm import LSTMRegressor
from src.models.mlp import MLPRegressor
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _strategy_specs(device: str) -> dict[str, dict]:
    return {
        "mlp": {
            "factory": lambda seed: MLPRegressor(
                hidden=(64, 32), dropout=0.3, epochs=40, batch_size=2048,
                lr=1e-3, weight_decay=1e-4, patience=4, val_frac=0.2,
                seed=seed, device=device,
            ),
            "target_col": "target_reg",
            "sequence_length": 1,
            "model_params": {
                "model": "MLP", "hidden": "(64,32)", "dropout": 0.3,
                "loss": "smooth_l1", "epochs": 40, "lr": 1e-3, "weight_decay": 1e-4,
            },
        },
        "lstm": {
            "factory": lambda seed: LSTMRegressor(
                hidden=32, num_layers=1, dropout=0.2, epochs=25, batch_size=2048,
                lr=1e-3, weight_decay=1e-4, patience=4, val_frac=0.2,
                seed=seed, device=device,
            ),
            "target_col": "target_reg",
            "sequence_length": 12,
            "model_params": {
                "model": "LSTM", "hidden": 32, "num_layers": 1, "dropout": 0.2,
                "sequence_length": 12, "loss": "smooth_l1", "epochs": 25, "lr": 1e-3,
            },
        },
    }


def _run_single_seed(
    strategy_name: str,
    seed: int,
    factory: Callable[[int], BaseModel],
    target_col: str,
    sequence_length: int,
    cfg: KaggleUSExperimentConfig,
    trading_cfg: TradingConfig,
    train_window_months: int,
    quantile: float,
) -> tuple[pd.DataFrame, dict, dict]:
    """Один прогон с фиксированным сидом. Возвращает (returns, metrics, paths)."""
    set_seed(seed)
    run_name = f"{strategy_name}_s{seed}"
    logger.info("--- %s | seed=%d | T=%d ---", strategy_name, seed, sequence_length)
    t0 = time.time()

    strategy = MLScoringStrategy(
        model_factory=lambda s=seed: factory(s),
        target_col=target_col,
        train_window_months=train_window_months,
        sequence_length=sequence_length,
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
    logger.info("seed %d done in %.1fs (Sharpe=%.3f, NAV=%.3f)", seed, elapsed, stats.sharpe, stats.final_nav)

    paths = save_run_artifacts(run_name, runner, stats, output_dir="results")

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
        "elapsed_sec": elapsed,
    }
    return runner.strategy_total_r.copy(), metrics, paths


def _aggregate_seeds(
    strategy_name: str,
    per_seed: list[dict],
    benchmark: pd.Series,
    output_dir: Path,
) -> dict:
    """Усреднить кривые/метрики по сидам, сохранить агрегированные артефакты."""
    out = output_dir / f"{strategy_name}_agg"
    out.mkdir(parents=True, exist_ok=True)

    returns_list = [r["returns"].iloc[:, 0].rename(f"seed_{r['seed']}") for r in per_seed]
    all_returns = pd.concat(returns_list, axis=1).fillna(0.0)
    mean_r = all_returns.mean(axis=1).rename("mean_total_r")
    std_r = all_returns.std(axis=1).rename("std_total_r")

    all_returns.to_parquet(out / "per_seed_returns.parquet")
    mean_r.to_frame().to_parquet(out / "mean_returns.parquet")

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"], **r["metrics"]})
    metrics_df = pd.DataFrame(rows).set_index("seed").sort_index()
    metrics_df.to_csv(out / "per_seed_metrics.csv")

    agg = metrics_df.agg(["mean", "std", "min", "max"]).T
    agg.to_csv(out / "agg_metrics.csv")

    fig_eq = plot_equity(mean_r, benchmark.reindex(mean_r.index).fillna(0),
                         strategy_name=f"{strategy_name} (mean of seeds)", benchmark_name="SPX (total)")
    fig_eq.savefig(out / "equity_mean.png", dpi=120)
    fig_dd = plot_drawdown(mean_r, strategy_name=f"{strategy_name} (mean)")
    fig_dd.savefig(out / "drawdown_mean.png", dpi=120)
    fig_rs = plot_rolling_sharpe(mean_r, window_days=252, strategy_name=f"{strategy_name} (mean)")
    fig_rs.savefig(out / "rolling_sharpe_mean.png", dpi=120)

    seed_curves = {f"seed_{r['seed']}": r["returns"].iloc[:, 0] for r in per_seed}
    fig_seeds = plot_comparison(seed_curves, benchmark, benchmark_name="SPX (total)",
                                title=f"{strategy_name}: per-seed equity")
    fig_seeds.savefig(out / "per_seed_equity.png", dpi=120)

    import matplotlib.pyplot as plt
    for fig in (fig_eq, fig_dd, fig_rs, fig_seeds):
        plt.close(fig)

    summary = {
        "strategy": strategy_name,
        "n_seeds": len(per_seed),
        "metrics_mean": agg["mean"].to_dict(),
        "metrics_std": agg["std"].to_dict(),
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Aggregated %d seeds for %s → %s", len(per_seed), strategy_name, out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["mlp", "lstm"], choices=["mlp", "lstm"])
    parser.add_argument("--seeds", type=int, default=5, help="Кол-во сидов на стратегию")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2017-11-09")
    parser.add_argument("--train-window", type=int, default=60)
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--broker-fee", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="dl-momentum-dl-reg")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = KaggleUSExperimentConfig()
    cfg.START_DATE = pd.Timestamp(args.start)
    cfg.END_DATE = pd.Timestamp(args.end)

    trading_cfg = TradingConfig(
        broker_fee=args.broker_fee, bid_ask_spread=0.0,
        total_exposure=1.0, trading_lag_days=1,
    )

    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        enabled=not args.no_mlflow,
    )

    specs = _strategy_specs(device=args.device)
    summaries = []
    for name in args.strategies:
        spec = specs[name]
        per_seed_results = []
        for seed in range(args.seeds):
            cfg_copy = KaggleUSExperimentConfig()
            cfg_copy.START_DATE = cfg.START_DATE
            cfg_copy.END_DATE = cfg.END_DATE
            run_name = f"{name}_s{seed}"
            params = {
                "strategy": name,
                "seed": seed,
                "train_window_months": args.train_window,
                "quantile": args.quantile,
                "rebal_freq": cfg_copy.REBALANCE_FREQ,
                "start": str(cfg.START_DATE.date()),
                "end": str(cfg.END_DATE.date()),
                "broker_fee": args.broker_fee,
                "device": args.device,
                **spec["model_params"],
            }
            try:
                returns, metrics, paths = _run_single_seed(
                    strategy_name=name, seed=seed,
                    factory=spec["factory"],
                    target_col=spec["target_col"],
                    sequence_length=spec["sequence_length"],
                    cfg=cfg_copy, trading_cfg=trading_cfg,
                    train_window_months=args.train_window, quantile=args.quantile,
                )
                with tracker.run(run_name, params=params) as t:
                    t.log_metrics(metrics)
                    for img in ("equity", "drawdown", "rolling_sharpe", "exposures"):
                        t.log_artifact(paths[img])
                    t.log_artifact(paths["metrics_csv"])
                per_seed_results.append({"seed": seed, "returns": returns, "metrics": metrics, "paths": paths})
            except Exception as e:
                logger.exception("seed %d for %s failed: %s", seed, name, e)

        if per_seed_results:
            bench_path = Path(f"results/{name}_s0/benchmark_returns.parquet")
            benchmark = pd.read_parquet(bench_path).iloc[:, 0] if bench_path.exists() else pd.Series(dtype=float)
            summary = _aggregate_seeds(name, per_seed_results, benchmark, Path("results"))
            summaries.append(summary)

    if summaries:
        logger.info("\n" + "=" * 70)
        logger.info("AGGREGATED SUMMARY")
        logger.info("=" * 70)
        for s in summaries:
            logger.info("%-12s n=%d  Sharpe=%.3f±%.3f  Final NAV=%.3f±%.3f  α=%.4f±%.4f  IR=%.3f±%.3f",
                        s["strategy"], s["n_seeds"],
                        s["metrics_mean"].get("sharpe", float("nan")),
                        s["metrics_std"].get("sharpe", float("nan")),
                        s["metrics_mean"].get("final_nav", float("nan")),
                        s["metrics_std"].get("final_nav", float("nan")),
                        s["metrics_mean"].get("alpha_benchmark", float("nan")),
                        s["metrics_std"].get("alpha_benchmark", float("nan")),
                        s["metrics_mean"].get("ir_benchmark", float("nan")),
                        s["metrics_std"].get("ir_benchmark", float("nan")))


if __name__ == "__main__":
    main()
