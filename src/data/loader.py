"""Загрузка Kaggle 'Huge Stock Market Dataset' в один tidy parquet.

Каждый файл `Stocks/<ticker>.us.txt` содержит CSV с колонками
`Date,Open,High,Low,Close,Volume,OpenInt`. Тикер выводим из имени файла.

Запуск как модуль:

    python -m src.data.loader [--config base.yaml] [--min-rows 252]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.io import load_config, write_parquet

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _ticker_from_filename(path: Path) -> str:
    """`aapl.us.txt` -> `AAPL`."""
    return path.stem.split(".")[0].upper()


def _read_one(path: Path, min_rows: int) -> pd.DataFrame | None:
    """Прочитать один файл; вернуть None, если он пустой/битый/слишком короткий."""
    try:
        if path.stat().st_size == 0:
            return None
        df = pd.read_csv(
            path,
            usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
            parse_dates=["Date"],
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError):
        return None

    if df.empty or len(df) < min_rows:
        return None

    df = df.rename(columns=str.lower)
    df["ticker"] = _ticker_from_filename(path)
    return df[OUTPUT_COLUMNS]


def load_kaggle_stocks(
    stocks_dir: Path | str,
    min_rows: int = 252,
    glob: str = "*.us.txt",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Прочитать все валидные файлы из `stocks_dir` в один long-формат DataFrame.

    Parameters
    ----------
    stocks_dir : каталог с файлами Kaggle (`.../Stocks/`).
    min_rows   : минимальное число дней истории, иначе тикер пропускается.
    glob       : маска имён файлов.
    show_progress : включить tqdm.

    Returns
    -------
    DataFrame со столбцами `[date, ticker, open, high, low, close, volume]`,
    отсортированный по `(ticker, date)`. `ticker` — `category` для экономии памяти.
    """
    stocks_dir = Path(stocks_dir)
    files = sorted(stocks_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No files matching '{glob}' in {stocks_dir}")

    parts: list[pd.DataFrame] = []
    iterator = tqdm(files, desc="Reading Kaggle Stocks") if show_progress else files
    for p in iterator:
        part = _read_one(p, min_rows)
        if part is not None:
            parts.append(part)

    if not parts:
        raise RuntimeError(f"No valid stock files in {stocks_dir} (min_rows={min_rows})")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["ticker"] = out["ticker"].astype("category")

    logger.info(
        "Loaded %d tickers (%d rows) from %s",
        out["ticker"].nunique(),
        len(out),
        stocks_dir,
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Load Kaggle stocks dataset to parquet")
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--min-rows", type=int, default=252,
                    help="минимум торговых дней в истории тикера")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["paths"]["raw_kaggle"]
    out_path = cfg["paths"]["processed_prices"]

    df = load_kaggle_stocks(raw_dir, min_rows=args.min_rows, show_progress=not args.no_progress)
    write_parquet(df, out_path)
    logger.info("Wrote %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
