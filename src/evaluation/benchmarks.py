"""Поиск файла бенчмарка (SPX total) для сводных отчётов.

Per-run артефакты сохраняют `benchmark_returns.parquet` в `results/{run_name}/`.
Для агрегатов (`mlp_agg`, …) бенчмарк лежит в одной из per-seed папок
(`mlp_s0`, …) — этот модуль находит первый доступный файл.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_benchmark_returns(results_dir: str | Path, strategy_base: str) -> pd.Series | None:
    """Загрузить SPX total returns для стратегии с базовым именем `strategy_base`.

    `strategy_base` — имя без суффикса `_agg` и без `_sN`, например ``"mlp"``,
    ``"ridge"``, ``"mc_dropout_filtered"``.
    """
    root = Path(results_dir)
    candidates = [
        root / strategy_base / "benchmark_returns.parquet",
        root / f"{strategy_base}_s0" / "benchmark_returns.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p).iloc[:, 0]
    matches = sorted(root.glob(f"{strategy_base}_s*/benchmark_returns.parquet"))
    if matches:
        return pd.read_parquet(matches[0]).iloc[:, 0]
    return None
