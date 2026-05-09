"""Подбор dropout для MLP при фиксированных hidden/epochs и прочих гиперах.

Базовый сценарий для текущего лучшего кандидата:
- hidden=(15,16)
- epochs=17
- sweep dropout: 0.0..0.6 (шаг 0.1)

Пример:
    PYTHONPATH=. python scripts/run_mlp_dropout_sweep.py
    PYTHONPATH=. python scripts/run_mlp_dropout_sweep.py --hidden 15x16 --epochs 17 --dropouts 0.1 0.2 0.3 0.4 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from quant_pml.config.trading_config import TradingConfig
from quant_pml.runner import build_backtest

from src.backtest.dataset_adapter import build_kaggle_dataset
from src.backtest.experiment_config import KaggleUSExperimentConfig
from src.backtest.strategy import MLScoringStrategy
from src.evaluation.artifacts import save_run_artifacts
from src.models.mlp import MLPRegressor
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _parse_hidden(spec: str) -> tuple[int, int]:
    s = spec.lower().strip().replace(" ", "")
    if "x" not in s:
        raise ValueError(f"Ожидание HxH2, например 15x16: {spec}")
    a, b = s.split("x", 1)
    return int(a), int(b)


def _factory(hidden: tuple[int, int], epochs: int, dropout: float, patience: int, device: str):
    def _make(seed: int) -> MLPRegressor:
        return MLPRegressor(
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            batch_size=2048,
            lr=1e-3,
            weight_decay=1e-4,
            patience=patience,
            val_frac=0.2,
            seed=seed,
            device=device,
        )

    return _make


def _run_one(
    run_name_base: str,
    hidden: tuple[int, int],
    epochs: int,
    dropout: float,
    seed: int,
    *,
    cfg: KaggleUSExperimentConfig,
    trading_cfg: TradingConfig,
    train_window_months: int,
    quantile: float,
    device: str,
    patience: int,
) -> dict[str, Any]:
    set_seed(seed)
    run_name = f"{run_name_base}_s{seed}"
    model_factory = _factory(hidden, epochs, dropout, patience, device)
    logger.info("=== %s hidden=%s epochs=%d dropout=%.2f ===", run_name, hidden, epochs, dropout)
    t0 = time.time()

    strategy = MLScoringStrategy(
        model_factory=lambda s=seed: model_factory(s),
        target_col="target_reg",
        train_window_months=train_window_months,
        sequence_length=1,
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
    save_run_artifacts(run_name, runner, stats, output_dir="results")

    return {
        "name": run_name_base,
        "hidden_h1": hidden[0],
        "hidden_h2": hidden[1],
        "epochs": epochs,
        "dropout": dropout,
        "seed": seed,
        "elapsed_sec": elapsed,
        "sharpe": stats.sharpe,
        "final_nav": stats.final_nav,
        "max_dd": stats.max_dd,
        "alpha_benchmark": stats.alpha_benchmark,
        "ir_benchmark": stats.ir_benchmark,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-oos", action="store_true")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--hidden", type=str, default="15x16")
    ap.add_argument("--epochs", type=int, default=17)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--dropouts", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--broker-fee", type=float, default=0.0005)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="results/mlp_dropout_sweep")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hidden = _parse_hidden(args.hidden)
    if args.start and args.end:
        start, end = args.start, args.end
    elif args.full_oos:
        start, end = "2010-01-01", "2017-11-09"
    else:
        start, end = "2016-06-01", "2017-06-30"

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_dates = KaggleUSExperimentConfig()
    cfg_dates.START_DATE = pd.Timestamp(start)
    cfg_dates.END_DATE = pd.Timestamp(end)

    trading_cfg = TradingConfig(
        broker_fee=args.broker_fee,
        bid_ask_spread=0.0,
        total_exposure=1.0,
        trading_lag_days=1,
    )

    meta = {"start": start, "end": end, "hidden": hidden, "epochs": args.epochs, "dropouts": args.dropouts}
    (out_root / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    h1, h2 = hidden
    for d in args.dropouts:
        base = f"mlp_h{h1}x{h2}_e{args.epochs}_d{str(round(d, 2)).replace('.', 'p')}"
        for seed in range(args.seeds):
            cfg_copy = KaggleUSExperimentConfig()
            cfg_copy.START_DATE = cfg_dates.START_DATE
            cfg_copy.END_DATE = cfg_dates.END_DATE
            try:
                row = _run_one(
                    base,
                    hidden,
                    args.epochs,
                    float(d),
                    seed,
                    cfg=cfg_copy,
                    trading_cfg=trading_cfg,
                    train_window_months=args.train_window,
                    quantile=args.quantile,
                    device=args.device,
                    patience=args.patience,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_root / "leaderboard_partial.csv", index=False)
            except Exception:
                logger.exception("FAILED %s seed=%s", base, seed)

    if not rows:
        logger.warning("No successful runs.")
        return

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(out_root / "leaderboard.csv", index=False)
    best = df.iloc[0]
    logger.info(
        "\nЛучший dropout=%.2f  Sharpe=%.4f NAV=%.4f (окно %s…%s)",
        float(best["dropout"]),
        float(best["sharpe"]),
        float(best["final_nav"]),
        start,
        end,
    )
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
