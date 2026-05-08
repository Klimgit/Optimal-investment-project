"""Сводный отчёт по всем стратегиям, сохранённым в `results/`.

Поддерживает два формата папок:

- **single-run** — `results/{strategy}/`: `total_returns.parquet` +
  `metrics.csv`. Используется для детерминированных стратегий (Ridge, LogReg).
- **multi-seed agg** — `results/{strategy}_agg/`: `mean_returns.parquet` +
  `agg_metrics.csv` (с колонкой `mean`). Используется для DL-стратегий с
  усреднением по сидам.

По умолчанию per-seed папки (`*_s0`, `*_s1`, ...) пропускаются и берутся
только агрегаты — это даёт по одной строке/линии на стратегию. Флаг
`--include-seeds` включает их.

Запуск:

    PYTHONPATH=. python scripts/compare_runs.py
    PYTHONPATH=. python scripts/compare_runs.py --include-seeds
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.benchmarks import load_benchmark_returns
from src.evaluation.plots import plot_comparison

logger = logging.getLogger(__name__)


def _summary_to_markdown(summary: pd.DataFrame) -> str:
    """Markdown-таблица без зависимости от `tabulate` (to_markdown)."""
    cols = list(summary.columns)
    parts = ["# Сводка стратегий\n\n", "| " + " | ".join(cols) + " |\n"]
    parts.append("|" + "|".join(["---"] * len(cols)) + "|\n")
    for _, row in summary.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and pd.notna(v):
                cells.append(f"{v:.4f}")
            elif v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            else:
                cells.append(str(v))
        parts.append("| " + " | ".join(cells) + " |\n")
    return "".join(parts)

KEY_METRICS = [
    "sharpe", "geom_avg_total_r", "geom_avg_xs_r", "std_xs_r", "max_dd",
    "alpha_benchmark", "alpha_benchmark_pvalue", "ir_benchmark",
    "alpha_buy_hold", "ir_buy_hold", "final_nav",
]

_SEED_RE = re.compile(r"_s\d+$")


def _load_run(run_dir: Path) -> dict | None:
    """Распознаёт single-run / multi-seed agg формат и возвращает унифицированный dict."""
    agg_metrics = run_dir / "agg_metrics.csv"
    mean_ret = run_dir / "mean_returns.parquet"
    if agg_metrics.exists() and mean_ret.exists():
        agg = pd.read_csv(agg_metrics, index_col=0)
        metrics = agg["mean"] if "mean" in agg.columns else agg.iloc[:, 0]
        returns = pd.read_parquet(mean_ret)
        # Нормализуем имя — без хвоста "_agg".
        name = run_dir.name.removesuffix("_agg")
        return {"name": name, "metrics": metrics, "returns": returns, "kind": "agg"}

    metrics_p = run_dir / "metrics.csv"
    returns_p = run_dir / "total_returns.parquet"
    if metrics_p.exists() and returns_p.exists():
        metrics = pd.read_csv(metrics_p, index_col=0)["value"]
        returns = pd.read_parquet(returns_p)
        return {"name": run_dir.name, "metrics": metrics, "returns": returns, "kind": "single"}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exclude", nargs="*", default=["_summary"])
    parser.add_argument("--include-seeds", action="store_true",
                        help="Включить в сравнение per-seed dirs (*_s0, *_s1, ...).")
    parser.add_argument("--markdown", type=str, default="",
                        help="Опционально записать сводную таблицу в Markdown-файл (путь).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.results_dir)
    candidates = []
    agg_strategy_names: set[str] = set()
    for p in sorted(root.iterdir()) if root.exists() else []:
        if not p.is_dir() or p.name in args.exclude:
            continue
        if p.name.endswith("_agg"):
            agg_strategy_names.add(p.name.removesuffix("_agg"))
        candidates.append(p)

    runs = []
    for p in candidates:
        if not args.include_seeds and _SEED_RE.search(p.name):
            base_name = _SEED_RE.sub("", p.name)
            if base_name in agg_strategy_names:
                continue
            if not args.include_seeds:
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
        row = {"strategy": r["name"], "kind": r["kind"]}
        for k in KEY_METRICS:
            try:
                row[k] = float(m.get(k))
            except (TypeError, ValueError):
                row[k] = None
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False, na_position="last")
    summary.to_csv(summary_dir / "metrics.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    bench = None
    for r in runs:
        bench = load_benchmark_returns(root, r["name"])
        if bench is not None:
            break

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_summary_to_markdown(summary), encoding="utf-8")
        logger.info("Wrote Markdown report: %s", md_path)

    strategies = {r["name"]: r["returns"].iloc[:, 0] for r in runs}
    fig = plot_comparison(strategies, bench, benchmark_name="SPX (total)")
    fig.savefig(summary_dir / "comparison.png", dpi=120)
    plt.close(fig)

    logger.info("Saved %s", summary_dir / "metrics.csv")
    logger.info("Saved %s", summary_dir / "comparison.png")


if __name__ == "__main__":
    main()
