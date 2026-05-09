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

Итоговый ``comparison.png``: ``mlp_best``, ``mlp_ab_newtrain_noregime`` (MLP
chrono+рег. без regime-фич), классические ML, follow, SPX и **Low-risk**
(``results/bench_low_vol_ls/`` при наличии). FF не выводятся.

Дополнительно (если есть соответствующие папки в ``results/``):

- ``_summary/comparison_ml_vs_dl.png`` — классические ML-бейзлайны vs DL / MLP;
- ``_summary/comparison_follow_fw_fl.png`` — Follow-the-winner vs Follow-the-loser;
- ``_summary/comparison_ml_dl_follow_all.png`` — те же три группы на одном графике.

Отключить: ``--no-extra-comparison-charts``.

Запуск:

    PYTHONPATH=. python scripts/compare_runs.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.benchmarks import load_benchmark_returns
from src.evaluation.plots import plot_comparison
from src.evaluation.run_registry import discover_run_directories, load_run as _load_run

logger = logging.getLogger(__name__)

                                                
_MAIN_CHART_LOW_VOL_LABEL = "Low-risk"

                                                                       
MAIN_SUMMARY_CHART_ORDER: list[str] = [
    "mlp_best",
    "mlp_ab_newtrain_noregime",
    "logreg",
    "ridge",
    "gbrt",
    "follow_winner",
    "follow_loser",
]


def _load_injected_low_vol(root: Path) -> pd.Series | None:
    path = root / "bench_low_vol_ls" / "total_returns.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path).iloc[:, 0]


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

                                                                                                     
CHART_ML: list[str] = ["ridge", "logreg", "gbrt"]
CHART_DL: list[str] = ["mlp_best", "mlp_v2", "mlp_agg", "mlp", "lstm"]
CHART_FOLLOW: list[str] = ["follow_winner", "follow_loser"]


def _curves_ordered(names: list[str], baseline: dict[str, pd.Series]) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for n in names:
        if n in baseline:
            out[n] = baseline[n]
    return out


def _dedupe_ordered(*chunks) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ch in chunks:
        for x in ch:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _save_theme_chart(
    baseline: dict[str, pd.Series],
    bench,
    summary_dir: Path,
    *,
    fname: str,
    title: str,
    name_order: list[str],
    min_curves: int = 1,
) -> None:
    curves = _curves_ordered(name_order, baseline)
    if len(curves) < min_curves:
        logger.info("Skip themed chart %s: only %d series (need >= %d)", fname, len(curves), min_curves)
        return
    fig = plot_comparison(curves, bench, benchmark_name="SPX (total)", dashed_labels=(), title=title)
    fig.savefig(summary_dir / fname, dpi=120, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    logger.info("Saved %s", summary_dir / fname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exclude", nargs="*", default=["_summary"])
    parser.add_argument("--only", nargs="*", default=[],
                        help="Оставить только эти стратегии (имена папок без _agg/_sN).")
    parser.add_argument("--include-seeds", action="store_true",
                        help="Включить в сравнение per-seed dirs (*_s0, *_s1, ...).")
    parser.add_argument("--markdown", type=str, default="",
                        help="Опционально записать сводную таблицу в Markdown-файл (путь).")
    parser.add_argument("--no-low-vol-line", action="store_true",
                        help="Не добавлять линию Low-risk из bench_low_vol_ls на comparison.png.")
    parser.add_argument(
        "--no-extra-comparison-charts",
        action="store_true",
        help="Не сохранять comparison_ml_vs_dl / comparison_follow_fw_fl / comparison_ml_dl_follow_all.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.results_dir)
    paired = discover_run_directories(
        root,
        exclude=args.exclude,
        only=args.only or None,
        include_seeds=args.include_seeds,
    )
    runs = [r for _, r in paired]

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

    baseline = {r["name"]: r["returns"].iloc[:, 0] for r in runs}

                                                                           
    plot_curves = _curves_ordered(MAIN_SUMMARY_CHART_ORDER, baseline)
    dashed_main: list[str] = []
    if not args.no_low_vol_line:
        inj = _load_injected_low_vol(root)
        if inj is not None:
            plot_curves[_MAIN_CHART_LOW_VOL_LABEL] = inj
            dashed_main.append(_MAIN_CHART_LOW_VOL_LABEL)
    if not plot_curves:
        logger.warning(
            "Main comparison chart: none of %s in results; plot may be empty (SPX only).",
            MAIN_SUMMARY_CHART_ORDER,
        )
    fig = plot_comparison(
        plot_curves,
        bench,
        benchmark_name="SPX (total)",
        dashed_labels=dashed_main,
        title="mlp_best, mlp_ab_newtrain_noregime, logreg, ridge, gbrt, follow vs SPX + low-risk",
    )
    fig.savefig(
        summary_dir / "comparison.png",
        dpi=120,
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)

    logger.info("Saved %s", summary_dir / "metrics.csv")
    logger.info("Saved %s", summary_dir / "comparison.png")

    if args.no_extra_comparison_charts:
        return

    ml_dl_names = _dedupe_ordered(CHART_ML, CHART_DL)
    _save_theme_chart(
        baseline,
        bench,
        summary_dir,
        fname="comparison_ml_vs_dl.png",
        title="ML-бейзлайны vs DL (кривые из текущего набора results)",
        name_order=ml_dl_names,
        min_curves=1,
    )
    _save_theme_chart(
        baseline,
        bench,
        summary_dir,
        fname="comparison_follow_fw_fl.png",
        title="Follow the winner vs follow the loser (r12 rule)",
        name_order=list(CHART_FOLLOW),
        min_curves=1,
    )
    merged = _dedupe_ordered(CHART_ML, CHART_DL, CHART_FOLLOW)
    _save_theme_chart(
        baseline,
        bench,
        summary_dir,
        fname="comparison_ml_dl_follow_all.png",
        title="ML + DL + follow rules (один график)",
        name_order=merged,
        min_curves=1,
    )


if __name__ == "__main__":
    main()
