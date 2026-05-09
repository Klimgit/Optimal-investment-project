"""Подбор числа эпох MLP (10…20) при фиксированных hidden и прочих гиперах.

Те же настройки, что у ``mlp`` в ``run_dl_reg.py``, кроме ``epochs`` (и
опционально ``patience``, см. флаги). По умолчанию hidden=(15,16) из последнего
подбора размерности.

    PYTHONPATH=. python scripts/run_mlp_epochs_sweep.py
    PYTHONPATH=. python scripts/run_mlp_epochs_sweep.py --full-oos
    PYTHONPATH=. python scripts/run_mlp_epochs_sweep.py --hidden 64x32 --epochs-min 10 --epochs-max 20
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
        msg = f"Ожидание HxH2, например 15x16: {spec}"
        raise ValueError(msg)
    a, b = s.split("x", 1)
    return int(a), int(b)


def _mlp_factory(hidden: tuple[int, int], epochs: int, patience: int, device: str):
    def _make(seed: int) -> MLPRegressor:
        return MLPRegressor(
            hidden=hidden,
            dropout=0.3,
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
    factory = _mlp_factory(hidden, epochs, patience, device)
    run_name = f"{run_name_base}_s{seed}"
    logger.info("=== %s hidden=%s epochs=%d patience=%d ===", run_name, hidden, epochs, patience)
    t0 = time.time()

    strategy = MLScoringStrategy(
        model_factory=lambda s=seed: factory(s),
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
        "patience": patience,
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
    ap.add_argument("--epochs-min", type=int, default=10)
    ap.add_argument("--epochs-max", type=int, default=20)
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (дефолт: min(4, max(1, epochs//3)) на каждый run)",
    )
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--broker-fee", type=float, default=0.0005)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="results/mlp_epochs_sweep")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.epochs_min > args.epochs_max:
        msg = "epochs-min must be <= epochs-max"
        raise ValueError(msg)

    hidden = _parse_hidden(args.hidden)
    epoch_list = list(range(args.epochs_min, args.epochs_max + 1))

    if args.start and args.end:
        start, end = args.start, args.end
    elif args.full_oos:
        start, end = "2010-01-01", "2017-11-09"
    else:
        start, end = "2015-07-01", "2017-06-30"

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

    meta = {
        "start": start,
        "end": end,
        "hidden": hidden,
        "epochs": epoch_list,
        "patience_mode": "fixed" if args.patience is not None else "scaled_per_epoch",
    }
    (out_root / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    h1, h2 = hidden
    for epochs in epoch_list:
        patience = args.patience if args.patience is not None else min(4, max(1, epochs // 3))
        base = f"mlp_h{h1}x{h2}_e{epochs}"
        for seed in range(args.seeds):
            cfg_copy = KaggleUSExperimentConfig()
            cfg_copy.START_DATE = cfg_dates.START_DATE
            cfg_copy.END_DATE = cfg_dates.END_DATE
            try:
                row = _run_one(
                    base, hidden, epochs, seed,
                    cfg=cfg_copy,
                    trading_cfg=trading_cfg,
                    train_window_months=args.train_window,
                    quantile=args.quantile,
                    device=args.device,
                    patience=patience,
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
        "\nЛучшие по Sharpe: epochs=%d  Sharpe=%.4f NAV=%.4f (окно %s…%s)",
        int(best["epochs"]),
        float(best["sharpe"]),
        float(best["final_nav"]),
        start,
        end,
    )
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
