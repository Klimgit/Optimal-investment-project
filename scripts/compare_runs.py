"""Сводный отчёт по всем стратегиям, сохранённым в `results/`.

Читает каждый `results/*/total_returns.parquet` + `metrics.csv`, собирает:
- сводную таблицу метрик (`results/_summary/metrics.csv`),
- общий equity-comparison график (`results/_summary/comparison.png`).

Запуск:

    PYTHONPATH=. python scripts/compare_runs.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.plots import plot_comparison

logger = logging.getLogger(__name__)

KEY_METRICS = [
    "sharpe", "geom_avg_total_r", "geom_avg_xs_r", "std_xs_r", "max_dd",
    "alpha_benchmark", "alpha_benchmark_pvalue", "ir_benchmark",
    "alpha_buy_hold", "ir_buy_hold", "final_nav",
]


def _load_run(run_dir: Path) -> dict | None:
    metrics_p = run_dir / "metrics.csv"
    returns_p = run_dir / "total_returns.parquet"
    if not (metrics_p.exists() and returns_p.exists()):
        return None
    metrics = pd.read_csv(metrics_p, index_col=0)["value"]
    returns = pd.read_parquet(returns_p)
    return {"name": run_dir.name, "metrics": metrics, "returns": returns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exclude", nargs="*", default=["_summary"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.results_dir)
    runs = []
    for p in sorted(root.iterdir()) if root.exists() else []:
        if not p.is_dir() or p.name in args.exclude:
            continue
        run = _load_run(p)
        if run is not None:
            runs.append(run)

    if not runs:
        logger.warning("No runs found in %s", root)
        return

    logger.info("Found %d runs: %s", len(runs), [r["name"] for r in runs])

    summary_dir = root / "_summary"
    summary_dir.mkdir(exist_ok=True)

    rows = []
    for r in runs:
        m = r["metrics"]
        row = {"strategy": r["name"]}
        for k in KEY_METRICS:
            try:
                row[k] = float(m.get(k))
            except (TypeError, ValueError):
                row[k] = None
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_dir / "metrics.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    bench = None
    bench_p = runs[0]["returns"]
    bench_path = root / runs[0]["name"] / "benchmark_returns.parquet"
    if bench_path.exists():
        bench = pd.read_parquet(bench_path).iloc[:, 0]

    strategies = {r["name"]: r["returns"].iloc[:, 0] for r in runs}
    fig = plot_comparison(strategies, bench, benchmark_name="SPX (total)")
    fig.savefig(summary_dir / "comparison.png", dpi=120)

    logger.info("Saved %s", summary_dir / "metrics.csv")
    logger.info("Saved %s", summary_dir / "comparison.png")


if __name__ == "__main__":
    main()
