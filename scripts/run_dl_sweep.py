"""Массовое сравнение DL-регрессоров (~15 конфигураций гиперпараметров).

Каждый конфиг — отдельный полный бэктест (как в `run_dl_reg.py`), имя сохраняется
в `results/{run_name}_s{seed}/`. В конце пишется `leaderboard.csv` в выходную папку.

Режим **quick** (по умолчанию для разумного времени): `2014-01-01`–`2017-11-09`, 1 seed.
Для полного OOS как в отчёте:

    PYTHONPATH=. python scripts/run_dl_sweep.py --full-oos --seeds 1

Запуск:

    PYTHONPATH=. python scripts/run_dl_sweep.py
    PYTHONPATH=. python scripts/run_dl_sweep.py --seeds 3 --device cpu
    PYTHONPATH=. python scripts/run_dl_sweep.py --list
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from quant_pml.config.trading_config import TradingConfig
from quant_pml.runner import build_backtest

from src.backtest.dataset_adapter import build_kaggle_dataset
from src.backtest.experiment_config import KaggleUSExperimentConfig
from src.backtest.strategy import MLScoringStrategy
from src.evaluation.artifacts import save_run_artifacts
from src.evaluation.plots import plot_comparison
from src.models.base import BaseModel
from src.models.lstm import LSTMRegressor
from src.models.mlp import MLPRegressor
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepEntry:
    name: str
    kind: str                  
    sequence_length: int
    factory: Callable[[int], BaseModel]
    params: dict[str, Any]


def _sweep_specs(device: str) -> list[SweepEntry]:
    """~15 конфигураций: MLP + LSTM, разный размер / dropout / lr / val / loss."""
    return [
        SweepEntry(
            "sw_mlp_64x32_d30",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(64, 32), dropout=0.3, epochs=40, batch_size=2048,
                lr=1e-3, weight_decay=1e-4, patience=4, val_frac=0.2,
                val_split_mode="random", grad_clip=None, scheduler=None,
                use_batch_norm=True, residual=False, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"hidden": "(64,32)", "dropout": 0.3, "lr": 1e-3, "val": "random"},
        ),
        SweepEntry(
            "sw_mlp_v2_128_res",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(128, 128), dropout=0.2, epochs=60, batch_size=1536,
                lr=7e-4, weight_decay=3e-4, patience=6, val_frac=0.2,
                val_split_mode="chrono", grad_clip=1.0, scheduler="cosine",
                use_batch_norm=True, residual=True, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"hidden": "(128,128)", "residual": True, "chrono": True, "sched": "cosine"},
        ),
        SweepEntry(
            "sw_mlp_256x64",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(256, 64), dropout=0.25, epochs=50, batch_size=2048,
                lr=8e-4, weight_decay=2e-4, patience=5, val_frac=0.2,
                val_split_mode="chrono", grad_clip=1.0, scheduler="cosine",
                use_batch_norm=True, residual=False, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"hidden": "(256,64)", "dropout": 0.25},
        ),
        SweepEntry(
            "sw_mlp_96x96_res",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(96, 96), dropout=0.2, epochs=55, batch_size=1536,
                lr=6e-4, weight_decay=3e-4, patience=6, val_frac=0.2,
                val_split_mode="chrono", grad_clip=1.0, scheduler="cosine",
                use_batch_norm=True, residual=True, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"hidden": "(96,96)", "residual": True},
        ),
        SweepEntry(
            "sw_mlp_hi_drop50",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(64, 32), dropout=0.5, epochs=45, batch_size=2048,
                lr=1e-3, weight_decay=1e-4, patience=5, val_frac=0.2,
                val_split_mode="random", grad_clip=None, scheduler=None,
                use_batch_norm=True, residual=False, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"dropout": 0.5},
        ),
        SweepEntry(
            "sw_mlp_lowlr_wd",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(64, 32), dropout=0.3, epochs=50, batch_size=2048,
                lr=3e-4, weight_decay=5e-4, patience=6, val_frac=0.2,
                val_split_mode="chrono", grad_clip=1.0, scheduler="cosine",
                use_batch_norm=True, residual=False, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"lr": 3e-4, "wd": 5e-4},
        ),
        SweepEntry(
            "sw_mlp_no_bn",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(128, 64), dropout=0.3, epochs=45, batch_size=2048,
                lr=1e-3, weight_decay=1e-4, patience=5, val_frac=0.2,
                val_split_mode="random", grad_clip=None, scheduler=None,
                use_batch_norm=False, residual=False, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"batch_norm": False},
        ),
        SweepEntry(
            "sw_mlp_small_bs",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(64, 32), dropout=0.3, epochs=45, batch_size=512,
                lr=1e-3, weight_decay=1e-4, patience=5, val_frac=0.2,
                val_split_mode="random", grad_clip=None, scheduler=None,
                use_batch_norm=True, residual=False, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"batch_size": 512},
        ),
        SweepEntry(
            "sw_mlp_mse",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(64, 32), dropout=0.3, epochs=45, batch_size=2048,
                lr=1e-3, weight_decay=1e-4, patience=5, val_frac=0.2,
                val_split_mode="random", grad_clip=None, scheduler=None,
                use_batch_norm=True, residual=False, loss="mse",
                seed=s, device=device,
            ),
            {"loss": "mse"},
        ),
        SweepEntry(
            "sw_mlp_ep80",
            "mlp",
            1,
            lambda s: MLPRegressor(
                hidden=(64, 32), dropout=0.3, epochs=80, batch_size=2048,
                lr=8e-4, weight_decay=1e-4, patience=8, val_frac=0.2,
                val_split_mode="chrono", grad_clip=1.0, scheduler="cosine",
                use_batch_norm=True, residual=False, loss="smooth_l1",
                seed=s, device=device,
            ),
            {"epochs": 80, "patience": 8},
        ),
                    
        SweepEntry(
            "sw_lstm_t12_h32",
            "lstm",
            12,
            lambda s: LSTMRegressor(
                hidden=32, num_layers=1, dropout=0.2, bidirectional=False,
                epochs=28, batch_size=2048, lr=1e-3, weight_decay=1e-4,
                patience=4, val_frac=0.2, seed=s, device=device,
            ),
            {"T": 12, "hidden": 32, "layers": 1},
        ),
        SweepEntry(
            "sw_lstm_t6_h32",
            "lstm",
            6,
            lambda s: LSTMRegressor(
                hidden=32, num_layers=1, dropout=0.2, bidirectional=False,
                epochs=28, batch_size=2048, lr=1e-3, weight_decay=1e-4,
                patience=4, val_frac=0.2, seed=s, device=device,
            ),
            {"T": 6, "hidden": 32},
        ),
        SweepEntry(
            "sw_lstm_t18_h32",
            "lstm",
            18,
            lambda s: LSTMRegressor(
                hidden=32, num_layers=1, dropout=0.2, bidirectional=False,
                epochs=30, batch_size=1536, lr=8e-4, weight_decay=1e-4,
                patience=5, val_frac=0.2, seed=s, device=device,
            ),
            {"T": 18, "hidden": 32},
        ),
        SweepEntry(
            "sw_lstm_t12_h64",
            "lstm",
            12,
            lambda s: LSTMRegressor(
                hidden=64, num_layers=1, dropout=0.25, bidirectional=False,
                epochs=30, batch_size=1536, lr=7e-4, weight_decay=1.5e-4,
                patience=5, val_frac=0.2, seed=s, device=device,
            ),
            {"T": 12, "hidden": 64},
        ),
        SweepEntry(
            "sw_lstm_t12_bi48",
            "lstm",
            12,
            lambda s: LSTMRegressor(
                hidden=48, num_layers=1, dropout=0.2, bidirectional=True,
                epochs=28, batch_size=1536, lr=8e-4, weight_decay=1e-4,
                patience=5, val_frac=0.2, seed=s, device=device,
            ),
            {"T": 12, "bi": True, "hidden": 48},
        ),
    ]


def _run_one(
    entry: SweepEntry,
    seed: int,
    cfg: KaggleUSExperimentConfig,
    trading_cfg: TradingConfig,
    train_window_months: int,
    quantile: float,
) -> dict[str, Any]:
    set_seed(seed)
    run_name = f"{entry.name}_s{seed}"
    logger.info("=== %s | kind=%s T=%d ===", run_name, entry.kind, entry.sequence_length)
    t0 = time.time()

    strategy = MLScoringStrategy(
        model_factory=lambda s=seed: entry.factory(s),
        target_col="target_reg",
        train_window_months=train_window_months,
        sequence_length=entry.sequence_length,
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

    paths = save_run_artifacts(run_name, runner, stats, output_dir="results")

    row = {
        "name": entry.name,
        "seed": seed,
        "kind": entry.kind,
        "sequence_length": entry.sequence_length,
        "elapsed_sec": elapsed,
        "sharpe": stats.sharpe,
        "geom_avg_total_r": stats.geom_avg_total_r,
        "std_xs_r": stats.std_xs_r,
        "max_dd": stats.max_dd,
        "alpha_benchmark": stats.alpha_benchmark,
        "alpha_benchmark_pvalue": stats.alpha_benchmark_pvalue,
        "ir_benchmark": stats.ir_benchmark,
        "final_nav": stats.final_nav,
        "params_json": json.dumps(entry.params, sort_keys=True),
    }
    logger.info(
        "%s Sharpe=%.4f NAV=%.4f α=%.4f IR=%.4f (%.1fs)",
        run_name,
        stats.sharpe,
        stats.final_nav,
        stats.alpha_benchmark,
        stats.ir_benchmark,
        elapsed,
    )
    _ = paths
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-oos", action="store_true", help="2010–2017-11-09 вместо quick 2014–2017")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--broker-fee", type=float, default=0.0005)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="results/dl_sweep_runs")
    ap.add_argument("--list", action="store_true", help="Показать имена конфигов и выйти")
    ap.add_argument("--only", nargs="*", help="Запустить только эти имена (подстрока или точное имя)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    specs = _sweep_specs(device=args.device)
    if args.list:
        for e in specs:
            print(e.name)
        return

    if args.start and args.end:
        start, end = args.start, args.end
    elif args.full_oos:
        start, end = "2010-01-01", "2017-11-09"
    else:
        start, end = "2014-01-01", "2017-11-09"

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.only:
        selected = []
        for e in specs:
            if any(e.name == o or o in e.name for o in args.only):
                selected.append(e)
        specs = selected
        if not specs:
            logger.error("No configs matched --only")
            return

    cfg_dates = KaggleUSExperimentConfig()
    cfg_dates.START_DATE = pd.Timestamp(start)
    cfg_dates.END_DATE = pd.Timestamp(end)

    trading_cfg = TradingConfig(
        broker_fee=args.broker_fee,
        bid_ask_spread=0.0,
        total_exposure=1.0,
        trading_lag_days=1,
    )

    meta = {
        "start": start,
        "end": end,
        "seeds": args.seeds,
        "train_window_months": args.train_window,
        "quantile": args.quantile,
        "n_configs": len(specs),
        "configs": [e.name for e in specs],
    }
    with (out_root / "sweep_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    rows: list[dict[str, Any]] = []
    for entry in specs:
        for seed in range(args.seeds):
            cfg_copy = KaggleUSExperimentConfig()
            cfg_copy.START_DATE = cfg_dates.START_DATE
            cfg_copy.END_DATE = cfg_dates.END_DATE
            try:
                row = _run_one(entry, seed, cfg_copy, trading_cfg, args.train_window, args.quantile)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_root / "leaderboard_partial.csv", index=False)
            except Exception:
                logger.exception("FAILED %s seed=%s", entry.name, seed)

    if not rows:
        logger.warning("No successful runs.")
        return

    lb = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    lb.to_csv(out_root / "leaderboard.csv", index=False)
    r0 = rows[0]
    bench_p = Path("results") / f"{r0['name']}_s{r0['seed']}" / "benchmark_returns.parquet"
    benchmark = pd.read_parquet(bench_p).iloc[:, 0] if bench_p.exists() else None
    curves = {}
    for r in rows:
        p = Path("results") / f"{r['name']}_s{r['seed']}" / "total_returns.parquet"
        if p.exists():
            curves[f"{r['name']}_s{r['seed']}"] = pd.read_parquet(p).iloc[:, 0]
    if len(curves) >= 2 and benchmark is not None:
        import matplotlib.pyplot as plt
        fig = plot_comparison(curves, benchmark, benchmark_name="SPX (total)", title="DL sweep")
        fig.savefig(out_root / "comparison_all.png", dpi=120)
        plt.close(fig)

    logger.info("Leaderboard written to %s", out_root / "leaderboard.csv")
    logger.info("\nTop 5 by Sharpe:\n%s", lb.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
