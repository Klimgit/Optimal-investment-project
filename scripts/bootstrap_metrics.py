"""Block-bootstrap доверительные интервалы по сохранённым дневным доходностям.

Примеры:

    PYTHONPATH=. python scripts/bootstrap_metrics.py results/mlp_best_agg
    PYTHONPATH=. python scripts/bootstrap_metrics.py results/mlp_best_agg \\
        --eval-start 2016-01-01 --paired --metrics all --json

Для IR и CAPM α нужен бенчмарк (файл рядом или ``*_s0/benchmark_returns.parquet``).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.block_bootstrap import block_bootstrap_metric_ci, block_bootstrap_paired_ci
from src.evaluation.holdout import resolve_benchmark_path, slice_period

logger = logging.getLogger(__name__)


def _load_strategy(run_dir: Path) -> pd.Series:
    for name in ("total_returns.parquet", "mean_returns.parquet"):
        p = run_dir / name
        if p.exists():
            return pd.read_parquet(p).iloc[:, 0].astype(float)
    raise FileNotFoundError(f"No total_returns/mean_returns in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for Sharpe / IR / alpha")
    parser.add_argument("run_dir", type=Path, help="e.g. results/mlp_best_agg")
    parser.add_argument("--metric", type=str, default="sharpe", help="legacy: sharpe only if --paired off")
    parser.add_argument(
        "--metrics",
        type=str,
        default="sharpe",
        help="При --paired: sharpe, ir_active, alpha_capm или all",
    )
    parser.add_argument("--paired", action="store_true", help="Парный bootstrap (strategy + SPX)")
    parser.add_argument("--eval-start", type=str, default="", help="Срез дат слева (holdout OOS)")
    parser.add_argument("--eval-end", type=str, default="", help="Срез дат справа")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block-len", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s = _load_strategy(args.run_dir)
    ev_s = args.eval_start.strip() or None
    ev_e = args.eval_end.strip() or None
    if ev_s or ev_e:
        (s,) = slice_period(s, start=ev_s, end=ev_e)

    out: dict = {"run_dir": str(args.run_dir)}
    if args.paired:
        bp = args.benchmark or resolve_benchmark_path(args.run_dir)
        if bp is None:
            raise SystemExit("Need benchmark parquet (--benchmark or adjacent *_s0)")
        b = pd.read_parquet(bp).iloc[:, 0].astype(float)
        if ev_s or ev_e:
            (b,) = slice_period(b, start=ev_s, end=ev_e)
        paired = block_bootstrap_paired_ci(
            s,
            b,
            rf=0.0,
            n_bootstrap=args.n_boot,
            block_len=args.block_len,
            alpha=args.alpha,
            random_state=args.seed,
        )
        want = args.metrics.strip().lower()
        if want == "all":
            out["bootstrap"] = paired
        else:
            keys = [x.strip() for x in want.split(",")]
            key_map = {
                "sharpe": "sharpe_ann_daily_xs",
                "ir_active": "ir_benchmark_daily_active",
                "alpha_capm": "alpha_capm_ann",
            }
            out["bootstrap"] = {}
            for k in keys:
                kk = key_map.get(k, k)
                if kk not in paired:
                    raise SystemExit(f"unknown metric key {k}")
                out["bootstrap"][kk] = paired[kk]
        out["benchmark"] = str(bp)
        if ev_s:
            out["eval_start"] = ev_s
        if ev_e:
            out["eval_end"] = ev_e
    else:
        if args.metric != "sharpe":
            raise SystemExit("Without --paired only --metric sharpe is supported")
        single = block_bootstrap_metric_ci(
            s,
            metric="sharpe",
            n_bootstrap=args.n_boot,
            block_len=args.block_len,
            alpha=args.alpha,
            random_state=args.seed,
        )
        out.update(single)
        out["run_dir"] = str(args.run_dir)

    if args.json:
        print(json.dumps(out, indent=2))
    elif args.paired and "bootstrap" in out:
        for name, d in out["bootstrap"].items():
            logger.info(
                "%s point=%.6f CI [%.6f, %.6f] n=%d",
                name,
                d["point"],
                d["ci_low"],
                d["ci_high"],
                int(d["n_days"]),
            )
    else:
        logger.info(
            "Sharpe point=%.4f  CI [%.4f, %.4f]",
            out["point"],
            out["ci_low"],
            out["ci_high"],
        )


if __name__ == "__main__":
    main()
