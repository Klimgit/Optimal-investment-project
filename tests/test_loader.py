"""Тесты loader.py на синтетических Kaggle-подобных файлах."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import _ticker_from_filename, _read_one, load_kaggle_stocks


def _write_kaggle_file(path: Path, n_rows: int, start: str = "2010-01-04") -> None:
    """Записать файл в Kaggle-формате `Date,Open,High,Low,Close,Volume,OpenInt`."""
    if n_rows == 0:
        path.write_text("")
        return
    dates = pd.bdate_range(start=start, periods=n_rows)
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": 10.0,
        "High": 11.0,
        "Low": 9.5,
        "Close": 10.5,
        "Volume": 1_000_000,
        "OpenInt": 0,
    })
    df.to_csv(path, index=False)


def test_ticker_from_filename():
    assert _ticker_from_filename(Path("aapl.us.txt")) == "AAPL"
    assert _ticker_from_filename(Path("brk-a.us.txt")) == "BRK-A"


def test_read_one_skips_empty(tmp_path: Path):
    p = tmp_path / "empty.us.txt"
    _write_kaggle_file(p, n_rows=0)
    assert _read_one(p, min_rows=10) is None


def test_read_one_skips_short(tmp_path: Path):
    p = tmp_path / "short.us.txt"
    _write_kaggle_file(p, n_rows=5)
    assert _read_one(p, min_rows=10) is None


def test_read_one_returns_correct_schema(tmp_path: Path):
    p = tmp_path / "good.us.txt"
    _write_kaggle_file(p, n_rows=20)
    df = _read_one(p, min_rows=10)
    assert df is not None
    assert list(df.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert (df["ticker"] == "GOOD").all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 20


def test_load_kaggle_stocks_concatenates_and_filters(tmp_path: Path):
    stocks = tmp_path / "Stocks"
    stocks.mkdir()
    _write_kaggle_file(stocks / "aaa.us.txt", n_rows=300)
    _write_kaggle_file(stocks / "bbb.us.txt", n_rows=300, start="2011-01-04")
    _write_kaggle_file(stocks / "tooshort.us.txt", n_rows=50)
    _write_kaggle_file(stocks / "empty.us.txt", n_rows=0)

    df = load_kaggle_stocks(stocks, min_rows=252, show_progress=False)
    tickers = sorted(df["ticker"].unique().tolist())
    assert tickers == ["AAA", "BBB"]
    assert len(df) == 600
    assert df.equals(df.sort_values(["ticker", "date"]).reset_index(drop=True))
    assert df["ticker"].dtype.name == "category"


def test_load_kaggle_stocks_raises_on_empty_dir(tmp_path: Path):
    stocks = tmp_path / "Stocks"
    stocks.mkdir()
    with pytest.raises(FileNotFoundError):
        load_kaggle_stocks(stocks, min_rows=10, show_progress=False)
