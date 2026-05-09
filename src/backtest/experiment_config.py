"""ExperimentConfig для нашего Kaggle-универса под `quant_pml`.

Минимально переопределяем `USExperimentConfig`:
- даты совпадают с нашим Kaggle-датасетом и OOS-периодом из configs/base.yaml;
- единственный фактор — `spx-rf` (есть только S&P 500 как бенчмарк);
- никаких HEDGING_ASSETS на первом этапе (хедж добавим позже);
- TARGETS пустой — таргеты живут внутри нашей стратегии (panel.parquet);
- RF_NAME = `rf` (используем нулевую безрисковую ставку, поскольку CRSP RF недоступен).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_pml.config.us_experiment_config import USExperimentConfig


@dataclass
class KaggleUSExperimentConfig(USExperimentConfig):
    """Конфиг под Kaggle Stocks dataset + S&P 500 (yfinance)."""

    PREFIX: str = field(default="kaggle_us_")

    DATA_PROCESSING_START_DATE: pd.Timestamp = field(default=pd.to_datetime("1995-01-01"))
    START_DATE: pd.Timestamp = field(default=pd.to_datetime("2010-01-01"))
    END_DATE: pd.Timestamp = field(default=pd.to_datetime("2017-11-09"))

    REBALANCE_FREQ: int | str | None = field(default="ME")
    HEDGE_FREQ: int | str | None = field(default=None)
    MIN_ROLLING_PERIODS: int | None = field(default=1)

                                        
    CAUSAL_WINDOW_SIZE: int | None = field(default=None)
    CAUSAL_WINDOW_END_DATE_FIELD: str | None = field(default=None)

    HEDGING_ASSETS: tuple[str, ...] = field(default=())
    FACTORS: tuple[str, ...] = field(default=("spx-rf",))
    TARGETS: tuple[str, ...] = field(default=())

    RF_NAME: str = field(default="rf")
    MKT_NAME: str = field(default="spx-rf")
