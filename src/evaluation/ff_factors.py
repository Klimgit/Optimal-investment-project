"""Загрузка ежемесячных факторов Fama–French 5 (2×3) и выравнивание на дневной индекс.

Используется для линий «size / value / quality» на графике сравнения:

- **SMB** — Small Minus Big (размер);
- **HML** — High Minus Low book-to-market (value);
- **RMW** — Robust Minus Weak profitability (прокси quality в FF5).

Источник: Kenneth French Data Library (ежемесячные доходности в %). Низкий
риск (low-vol) в классическом FF5 нет — для него см. бенчмарк в нашем
universe (`scripts/run_factor_benchmarks.py`).

Низкоуровневое преобразование месячной простой доходности R в дневные ставки
с тем же накоплением за месяц: (1+R)^{1/N}-1 на каждый из N торговых дней.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Final
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

FF5_ZIP_URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)


def fetch_ff_factors_5x2(cache_path: Path | None = None) -> pd.DataFrame:
    """Скачать (или прочитать из кеша) FF5 2×3, месячные простые доходности.

    Колонки: ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``, ``RF`` (в долях, не в %).

    Индекс: конец месяца (DatetimeIndex normalized to month-end).
    """
    if cache_path is not None and cache_path.exists():
        return _read_saved_monthly(cache_path)

    logger.info("Downloading Fama-French 5 factors from %s", FF5_ZIP_URL)
    with urlopen(FF5_ZIP_URL, timeout=120) as resp:                                   
        raw = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    text = zf.read(csv_name).decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()]

    hdr_i = next(i for i, ln in enumerate(lines) if "Mkt-RF" in ln and "SMB" in ln)
    cols_use = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    rows: list[list] = []
    for ln in lines[hdr_i + 1 :]:
        if not ln or ln.startswith("COPYRIGHT") or ln.startswith("Copyright"):
            break
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7:
            continue
        try:
            ym = int(parts[0])
        except ValueError:
            continue
        try:
            floats = [float(parts[j]) for j in range(1, 7)]
        except ValueError:
            continue
        rows.append([ym] + floats)

    tbl = pd.DataFrame(rows, columns=["ym", *cols_use])
    tbl = tbl.dropna(subset=["Mkt-RF"])

    idx = pd.to_datetime(tbl["ym"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    out = tbl[cols_use].div(100.0)
    out.index = idx

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
        logger.info("Cached FF factors to %s", cache_path)

    return out


def _read_saved_monthly(cache_path: Path) -> pd.DataFrame:
    if cache_path.suffix == ".parquet":
        return pd.read_parquet(cache_path)
    return pd.read_csv(cache_path, index_col=0, parse_dates=True)


def monthly_factor_to_daily_simple(
    monthly: pd.Series,
    daily_index: pd.DatetimeIndex,
) -> pd.Series:
    """Переразложить месячную простую доходность `monthly` на торговые дни.

    Для каждого календарного месяца: все дни из ``daily_index`` в этом месяце
    получают одинаковую простую дневную доходность d такую, что
    ``(1+d)^N - 1 = R_month`` при N торговых днях в месяце.
    """
    monthly = monthly.sort_index().astype("float64")
    di = daily_index.unique().sort_values()
    per = pd.Series([pd.Timestamp(x).to_period("M") for x in di], index=di)
    out = pd.Series(0.0, index=di, dtype="float64")

    for ts, rm in monthly.items():
        target = pd.Timestamp(ts).to_period("M")
        bd = di[per.values == target]
        n = len(bd)
        if n == 0 or not pd.notna(rm):
            continue
        d_day = float((1.0 + rm) ** (1.0 / n) - 1.0)
        out.loc[bd] = d_day

    return out.rename(monthly.name or "factor")
