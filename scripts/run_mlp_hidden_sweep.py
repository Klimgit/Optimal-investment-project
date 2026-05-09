"""Подбор размерности двух скрытых слоёв MLP при фиксированных прочих гиперах.

Все конфигурации совпадают с «простым» baseline из ``run_dl_reg`` (``mlp``):
epochs=40, dropout=0.3, lr, weight_decay, batch_size и т.д. Меняется только
пара ``hidden=(h1,h2)``.

Результат: артефакты в ``results/<name>_s{N}/``, сводная ``leaderboard.csv`` и
вывод топа по Sharpe (на коротком окне — ориентир; финал лучше на полном OOS).

    PYTHONPATH=. python scripts/run_mlp_hidden_sweep.py
    PYTHONPATH=. python scripts/run_mlp_hidden_sweep.py --full-oos
    PYTHONPATH=. python scripts/run_mlp_hidden_sweep.py --sizes 64x32 128x96 96x96
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

                                                                                    
DEFAULT_HIDDEN_GRID: list[tuple[int, int]] = [
    (15, 16),
    (32, 16),
    (48, 24),
    (64, 32),
    (64, 64),
    (80, 40),
    (96, 48),
    (128, 64),
    (128, 96),
    (128, 128),
    (160, 80),
]


def _baseline_mlp_factory(hidden: tuple[int, int], device: str):
    """Те же настройки, что у ключей ``mlp`` в ``scripts/run_dl_reg.py``."""

    def _make(seed: int) -> MLPRegressor:
        return MLPRegressor(
            hidden=hidden,
            dropout=0.3,
            epochs=40,
            batch_size=2048,
            lr=1e-3,
            weight_decay=1e-4,
            patience=4,
            val_frac=0.2,
            seed=seed,
            device=device,
        )

    return _make


def _hidden_to_name(hidden: tuple[int, int]) -> str:
    a, b = hidden
    return f"mlp_h{a}x{b}"


def _parse_sizes_specs(specs: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in specs:
        s = s.lower().strip().replace(" ", "")
        if "x" not in s:
            msg = f"Ожидание формата HxH2, например 128x64, получено: {s}"
            raise ValueError(msg)
        a, b = s.split("x", 1)
        out.append((int(a), int(b)))
    return out


def _run_one(
    run_name_base: str,
    hidden: tuple[int, int],
    seed: int,
    *,
    cfg: KaggleUSExperimentConfig,
    trading_cfg: TradingConfig,
    train_window_months: int,
    quantile: float,
    device: str,
) -> dict[str, Any]:
    set_seed(seed)
    factory = _baseline_mlp_factory(hidden, device)
    run_name = f"{run_name_base}_s{seed}"
    logger.info("=== %s hidden=%s ===", run_name, hidden)
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

    row = {
        "name": run_name_base,
        "hidden_h1": hidden[0],
        "hidden_h2": hidden[1],
        "params": str(hidden),
        "seed": seed,
        "elapsed_sec": elapsed,
        "sharpe": stats.sharpe,
        "final_nav": stats.final_nav,
        "max_dd": stats.max_dd,
        "alpha_benchmark": stats.alpha_benchmark,
        "ir_benchmark": stats.ir_benchmark,
    }
    logger.info("%s Sharpe=%.4f NAV=%.4f (%.1fs)", run_name, stats.sharpe, stats.final_nav, elapsed)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-oos", action="store_true")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--sizes", nargs="*", default=[], help='Например: 64x32 128x64 (иначе дефолтная сетка)')
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--broker-fee", type=float, default=0.0005)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="results/mlp_hidden_sweep")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hidden_grid = _parse_sizes_specs(args.sizes) if args.sizes else DEFAULT_HIDDEN_GRID

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

    meta = {"start": start, "end": end, "hidden_grid": hidden_grid, "baseline_note": "same as run_dl_reg mlp except hidden"}
    (out_root / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for hidden in hidden_grid:
        base = _hidden_to_name(hidden)
        for seed in range(args.seeds):
            cfg_copy = KaggleUSExperimentConfig()
            cfg_copy.START_DATE = cfg_dates.START_DATE
            cfg_copy.END_DATE = cfg_dates.END_DATE
            try:
                row = _run_one(
                    base, hidden, seed,
                    cfg=cfg_copy,
                    trading_cfg=trading_cfg,
                    train_window_months=args.train_window,
                    quantile=args.quantile,
                    device=args.device,
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
        "\nЛучшие по Sharpe (на окне %s…%s, ориентир): %sx%s  Sharpe=%.4f NAV=%.4f",
        start, end, int(best["hidden_h1"]), int(best["hidden_h2"]),
        float(best["sharpe"]), float(best["final_nav"]),
    )
    logger.info("\n%s", df.to_string(index=False))
    logger.info("Полное сравнение: зафиксируйте top-3 на --full-oos и выберите по IR/просадке.")


if __name__ == "__main__":
    main()
