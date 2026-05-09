"""Подбор batch_size, weight_decay, grad_clip при фиксированной архитектуре ``mlp_best``.

Фиксируется (по умолчанию то, что мы уже подобрали):
- hidden=(15,16), epochs=17, dropout=0.4, lr=1e-3, patience=4, random val
- scheduler=cosine (как у дефолтного ``MLPRegressor`` в коде)

Пример:

    PYTHONPATH=. python scripts/run_mlp_train_kw_sweep.py
    PYTHONPATH=. python scripts/run_mlp_train_kw_sweep.py --full-oos
    PYTHONPATH=. python scripts/run_mlp_train_kw_sweep.py --batch-sizes 512 1024 --wds 1e-4 3e-4 --grad-clips none 1.0
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


def _parse_grad_clips(specs: list[str]) -> list[float | None]:
    out: list[float | None] = []
    for s in specs:
        t = s.strip().lower()
        if t in ("none", "null", "off", "0", "no"):
            out.append(None)
        else:
            out.append(float(s))
    return out


def _wd_tag(wd: float) -> str:
    if wd >= 0.01:
        return f"{wd:.0e}".replace("e-0", "em").replace("e+", "ep")
    return str(wd).replace(".", "p")


def _gc_tag(gc: float | None) -> str:
    if gc is None:
        return "gcn"
    return f"gc{str(gc).replace('.', 'p')}"


def _factory(
    *,
    hidden: tuple[int, int],
    epochs: int,
    dropout: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float | None,
    scheduler: str | None,
    val_split_mode: str,
    device: str,
):
    def _make(seed: int) -> MLPRegressor:
        return MLPRegressor(
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            val_frac=0.2,
            val_split_mode=val_split_mode,
            grad_clip=grad_clip,
            scheduler=scheduler,
            seed=seed,
            device=device,
        )

    return _make


def _run_one(
    run_name_base: str,
    *,
    seed: int,
    model_factory,
    cfg: KaggleUSExperimentConfig,
    trading_cfg: TradingConfig,
    train_window_months: int,
    quantile: float,
    meta_row: dict[str, Any],
) -> dict[str, Any]:
    set_seed(seed)
    run_name = f"{run_name_base}_s{seed}"
    logger.info("=== %s ===", run_name)
    for k, v in meta_row.items():
        logger.info("  %s=%s", k, v)

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
        "seed": seed,
        **meta_row,
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
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        help='"cosine" | "none" (отключает scheduler)',
    )
    ap.add_argument("--val-split-mode", type=str, default="random")
    ap.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048],
        help='Развернутая сетка: `--batch-sizes 512 1024 1536 2048`',
    )
    ap.add_argument(
        "--wds",
        nargs="+",
        type=float,
        default=[5e-5, 1e-4, 3e-4],
        help='Добавьте 1e-3 и т.д. при необходимости.',
    )
    ap.add_argument(
        "--grad-clips",
        nargs="+",
        type=str,
        default=["none", "1.0"],
        help='Число или none/off. Полная решётка: none 0.5 1.0 2.0',
    )
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--broker-fee", type=float, default=0.0005)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="results/mlp_train_kw_sweep")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hidden = _parse_hidden(args.hidden)
    grad_clips = _parse_grad_clips(args.grad_clips)
    sched = None if args.scheduler.lower() == "none" else args.scheduler

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

    meta = {
        "start": start,
        "end": end,
        "hidden": hidden,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "lr": args.lr,
        "patience": args.patience,
        "scheduler": sched,
        "val_split_mode": args.val_split_mode,
        "batch_sizes": args.batch_sizes,
        "weight_decays": args.wds,
        "grad_clips": [None if g is None else float(g) for g in grad_clips],
    }
    (out_root / "sweep_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    h1, h2 = hidden
    for bs in args.batch_sizes:
        for wd in args.wds:
            for gc in grad_clips:
                base = f"mlp_h{h1}x{h2}_e{args.epochs}_d{str(args.dropout).replace('.', 'p')}_bs{bs}_wd{_wd_tag(wd)}_{_gc_tag(gc)}"
                mf = _factory(
                    hidden=hidden,
                    epochs=args.epochs,
                    dropout=args.dropout,
                    batch_size=bs,
                    lr=args.lr,
                    weight_decay=wd,
                    patience=args.patience,
                    grad_clip=gc,
                    scheduler=sched,
                    val_split_mode=args.val_split_mode,
                    device=args.device,
                )
                mrow = {
                    "hidden_h1": h1,
                    "hidden_h2": h2,
                    "epochs": args.epochs,
                    "dropout": args.dropout,
                    "batch_size": bs,
                    "weight_decay": wd,
                    "grad_clip": gc if gc is not None else "none",
                }
                for seed in range(args.seeds):
                    cfg_copy = KaggleUSExperimentConfig()
                    cfg_copy.START_DATE = cfg_dates.START_DATE
                    cfg_copy.END_DATE = cfg_dates.END_DATE
                    try:
                        row = _run_one(
                            base,
                            seed=seed,
                            model_factory=mf,
                            cfg=cfg_copy,
                            trading_cfg=trading_cfg,
                            train_window_months=args.train_window,
                            quantile=args.quantile,
                            meta_row=mrow,
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
    b = df.iloc[0]
    logger.info(
        "\nЛучшие: bs=%s wd=%s grad_clip=%s  Sharpe=%.4f NAV=%.4f",
        int(b["batch_size"]),
        b["weight_decay"],
        b["grad_clip"],
        float(b["sharpe"]),
        float(b["final_nav"]),
    )
    logger.info("\n%s", df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
