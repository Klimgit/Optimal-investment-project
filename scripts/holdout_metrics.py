"""Точечные holdout-метрики (Sharpe / IR / CAPM α) на календарном срезе без повторного бэктеста."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.holdout import (
    DEFAULT_EVAL_START,
    collect_holdout_metrics,
    resolve_benchmark_path,
    write_holdout_json,
)

logger = logging.getLogger(__name__)


def _load_strategy(run_dir: Path) -> pd.Series:
    for name in ("total_returns.parquet", "mean_returns.parquet"):
        p = run_dir / name
        if p.exists():
            return pd.read_parquet(p).iloc[:, 0].astype(float)
    raise FileNotFoundError(f"No returns parquet in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--eval-start",
        type=str,
        default=str(DEFAULT_EVAL_START.date()),
        help="Начало финального OOS (включительно)",
    )
    parser.add_argument("--eval-end", type=str, default="", help="Конец среза (опционально)")
    parser.add_argument("--benchmark", type=Path, default=None, help="Явный путь к benchmark_returns.parquet")
    parser.add_argument("--write", action="store_true", help="Сохранить metrics_holdout.json")
    parser.add_argument("--json", action="store_true", help="Печать JSON в stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    strat = _load_strategy(args.run_dir)
    bench_path = args.benchmark or resolve_benchmark_path(args.run_dir)
    if bench_path is None:
        raise SystemExit(
            "benchmark_returns.parquet not found; pass --benchmark path "
            "(for *_agg use e.g. results/mlp_best_s0/benchmark_returns.parquet)"
        )
    bench = pd.read_parquet(bench_path).iloc[:, 0].astype(float)

    ev_end = args.eval_end.strip() or None
    metrics = collect_holdout_metrics(
        strat,
        bench,
        eval_start=args.eval_start,
        eval_end=ev_end,
        rf=0.0,
    )
    metrics["benchmark_source"] = str(bench_path)
    metrics["protocol"] = {
        "hyper_tuning_end": "2015-12-31",
        "eval_start_default": str(DEFAULT_EVAL_START.date()),
    }

    if args.write:
        p = write_holdout_json(args.run_dir, metrics)
        logger.info("Wrote %s", p)
    if args.json:
        print(json.dumps(metrics, indent=2))
    elif not args.write:
        logger.info("%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
