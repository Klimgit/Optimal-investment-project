"""Скачать дневные котировки `^GSPC` (S&P 500) через yfinance в parquet.

Запуск:

    python scripts/download_spx.py [--config base.yaml] [--start ...] [--end ...]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.io import load_config, write_parquet

logger = logging.getLogger(__name__)


def download_spx(start: str, end: str | None, out_path: Path | str) -> pd.DataFrame:
    raw = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance вернул пустой DataFrame для ^GSPC")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns and "adjclose" in df.columns:
        df = df.rename(columns={"adjclose": "adj_close"})
    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"]
            if c in df.columns]
    df = df[keep].sort_values("date").reset_index(drop=True)
    write_parquet(df, out_path)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download SPX (^GSPC) via yfinance")
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    start = args.start or cfg["dates"]["start"]
    end = args.end or cfg["dates"]["oos_end"]
    out = cfg["paths"]["benchmark"]

    df = download_spx(start, end, out)
    logger.info("Saved %d rows of SPX (%s..%s) to %s",
                len(df), df["date"].min().date(), df["date"].max().date(), out)


if __name__ == "__main__":
    main()
