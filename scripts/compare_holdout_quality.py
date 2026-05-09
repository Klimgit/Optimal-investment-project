"""Сравнение стратегий на holdout-OOS (точечные метрики + bootstrap ДИ).

Использует те же папки ``results/``, что ``compare_runs.py``. По умолчанию окно
отчёта: ``2016-01-01`` … конец ряда (см. ``docs/holdout_protocol.md``).

Пример:

    PYTHONPATH=. python scripts/compare_holdout_quality.py
    PYTHONPATH=. python scripts/compare_holdout_quality.py --impact-eta 1e-4 --n-boot 600
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.block_bootstrap import block_bootstrap_paired_ci
from src.evaluation.holdout import (
    DEFAULT_EVAL_START,
    collect_holdout_metrics,
    resolve_benchmark_path,
    slice_period,
)
from src.evaluation.market_impact import approximate_daily_turnover_fraction, linear_impact_haircut
from src.evaluation.run_registry import discover_run_directories, resolve_rebal_weights_path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exclude", nargs="*", default=["_summary"])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--include-seeds", action="store_true")
    parser.add_argument("--eval-start", type=str, default=str(DEFAULT_EVAL_START.date()))
    parser.add_argument("--eval-end", type=str, default="")
    parser.add_argument("--n-boot", type=int, default=600)
    parser.add_argument("--block-len", type=int, default=21)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--impact-eta",
        type=float,
        default=None,
        help="Если задано: вычитание η·turnover (оборот из rebal_weights), повтор метрик на скорректированных доходностях.",
    )
    parser.add_argument("--no-bootstrap", action="store_true", help="Только точечные оценки")
    parser.add_argument("--markdown", type=str, default="", help="Путь для отчёта .md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.results_dir)
    pairs = discover_run_directories(
        root,
        exclude=args.exclude,
        only=args.only or None,
        include_seeds=args.include_seeds,
    )

    ev_end = args.eval_end.strip() or None
    rows: list[dict] = []

    min_holdout_days = 30

    for run_dir, run in pairs:
        strat_raw = run["returns"].iloc[:, 0].astype(float)
        bp = resolve_benchmark_path(run_dir)
        if bp is None:
            logger.warning("Skip %s: no benchmark_returns", run_dir.name)
            continue
        bench_raw = pd.read_parquet(bp).iloc[:, 0].astype(float)
        aligned = pd.concat([strat_raw.rename("s"), bench_raw.rename("b")], axis=1).dropna()
        if len(aligned) < min_holdout_days:
            logger.warning(
                "Skip %s: only %d overlapping days with benchmark (need >= %d)",
                run_dir.name,
                len(aligned),
                min_holdout_days,
            )
            continue
        strat, bench = aligned["s"], aligned["b"]

        base = collect_holdout_metrics(
            strat,
            bench,
            eval_start=args.eval_start,
            eval_end=ev_end,
            rf=0.0,
        )
        if base["n_days"] < min_holdout_days:
            logger.warning(
                "Skip %s: holdout window %s..%s has only %d days (need >= %d)",
                run["name"],
                base["eval_start"],
                base["eval_end"],
                base["n_days"],
                min_holdout_days,
            )
            continue

        row = {
            "strategy": run["name"],
            "kind": run["kind"],
            "folder": run_dir.name,
            "eval_start": base["eval_start"],
            "eval_end": base["eval_end"],
            "n_days": base["n_days"],
            "sharpe_ho": base["sharpe_ann_daily_xs"],
            "ir_ho": base["ir_benchmark_daily_active"],
            "alpha_capm_ann_ho": base["alpha_capm_ann"],
        }

        st, bt = slice_period(strat, bench, start=args.eval_start, end=ev_end)
        if not args.no_bootstrap:
            try:
                ci = block_bootstrap_paired_ci(
                    st,
                    bt,
                    rf=0.0,
                    n_bootstrap=args.n_boot,
                    block_len=args.block_len,
                    alpha=args.bootstrap_alpha,
                    random_state=args.seed,
                )
                row["sharpe_ci_low"] = ci["sharpe_ann_daily_xs"]["ci_low"]
                row["sharpe_ci_high"] = ci["sharpe_ann_daily_xs"]["ci_high"]
                row["ir_ci_low"] = ci["ir_benchmark_daily_active"]["ci_low"]
                row["ir_ci_high"] = ci["ir_benchmark_daily_active"]["ci_high"]
                row["alpha_ci_low"] = ci["alpha_capm_ann"]["ci_low"]
                row["alpha_ci_high"] = ci["alpha_capm_ann"]["ci_high"]
            except Exception as e:
                logger.warning("Bootstrap failed for %s: %s", run["name"], e)

        if args.impact_eta is not None:
            rw = resolve_rebal_weights_path(run_dir)
            if rw is None:
                row["sharpe_ho_impact"] = None
                row["note_impact"] = "no_rebal_weights"
            else:
                rebal = pd.read_parquet(rw)
                turn = approximate_daily_turnover_fraction(rebal, strat_raw.index)
                strat_i = linear_impact_haircut(strat_raw, turn, eta=args.impact_eta)
                ai = pd.concat([strat_i.rename("s"), bench_raw.rename("b")], axis=1).dropna()
                if len(ai) >= min_holdout_days:
                    imp = collect_holdout_metrics(
                        ai["s"],
                        ai["b"],
                        eval_start=args.eval_start,
                        eval_end=ev_end,
                        rf=0.0,
                    )
                    row["sharpe_ho_impact"] = imp["sharpe_ann_daily_xs"]
                else:
                    row["sharpe_ho_impact"] = None
                    row["note_impact"] = "impact_align_short"
                row["impact_eta"] = args.impact_eta

        rows.append(row)

    if not rows:
        logger.warning("Nothing to compare.")
        return

    df = pd.DataFrame(rows).sort_values("sharpe_ho", ascending=False, na_position="last")
    summary_dir = root / "_summary"
    summary_dir.mkdir(exist_ok=True)
    out_csv = summary_dir / "holdout_quality.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d strategies)", out_csv, len(df))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    payload = {
        "eval_start": args.eval_start,
        "eval_end": ev_end,
        "n_boot": args.n_boot,
        "impact_eta": args.impact_eta,
        "rows": rows,
    }
    (summary_dir / "holdout_quality.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Holdout quality comparison\n\n",
            f"Окно: **{args.eval_start}** — **{ev_end or 'конец ряда'}**. ",
            f"Bootstrap: **{args.n_boot}** replicates (block {args.block_len}d). ",
        ]
        if args.impact_eta is not None:
            lines.append(f"Impact η={args.impact_eta} (линейный haircut по обороту из весов).\n\n")
        else:
            lines.append("\n\n")
        cols = list(df.columns)
        lines.append("| " + " | ".join(cols) + " |\n")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float) and pd.notna(v):
                    cells.append(f"{v:.4f}")
                elif v is None or (isinstance(v, float) and pd.isna(v)):
                    cells.append("")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |\n")
        md_path.write_text("".join(lines), encoding="utf-8")
        logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
