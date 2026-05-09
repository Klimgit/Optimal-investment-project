"""Детерминированный скоринг по одному столбцу панели (без обучения модели).

Тот же L/S топ/низкий квантиль и равные веса, что у ``MLScoringStrategy``,
но ``score_i = score_sign * panel[snap_date, ticker, score_column]``.
Используется для простых факторных бенчмарков (напр. низкая волатильность по
``sigma_ann`` после кросс-секционного Z-score).

Согласовано с методологией со слайдов: помесячный walk-forward через quant_pml,
без параметрического ML-слоя.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pml.strategies.ml.scoring.base_ml_scoring_strategy import BaseMLScoringStrategy
from quant_pml.strategies.optimization_data import PredictionData, TrainingData


class FactorColumnStrategy(BaseMLScoringStrategy):
    """Скор из одной колонки ``panel.parquet`` на последнем месячном снимке ≤ pred."""

    def __init__(
        self,
        score_column: str,
        *,
        panel_path: str | Path = "data/features/panel.parquet",
        score_sign: float = 1.0,
        mode: str = "long_short",
        quantile: float = 0.1,
        weighting_scheme: str = "equally_weighted",
    ) -> None:
        super().__init__(
            mode=mode,
            quantile=quantile,
            weighting_scheme=weighting_scheme,
        )
        self.score_column = score_column
        self.score_sign = score_sign
        self.panel_path = Path(panel_path)
        self._panel: pd.DataFrame | None = None

    def _load_panel(self) -> pd.DataFrame:
        if self._panel is None:
            df = pd.read_parquet(self.panel_path)
            df["ticker"] = df["ticker"].astype(str)
            df["date"] = pd.to_datetime(df["date"])
            self._panel = df.set_index(["date", "ticker"]).sort_index()
        return self._panel

    def _fit(self, training_data: TrainingData) -> None:                
        pass

    def predict_scores(self, prediction_data: PredictionData) -> pd.Series:
        panel = self._load_panel()
        pred_date = pd.Timestamp(prediction_data.pred_date)

        snap_dates = panel.index.get_level_values("date").unique().sort_values()
        valid = snap_dates[snap_dates <= pred_date]
        if len(valid) == 0:
            return pd.Series(dtype=float, name="score")
        snap_date = valid.max()

        try:
            col = panel.loc[snap_date, self.score_column]
        except KeyError:
            return pd.Series(dtype=float, name="score")

        if isinstance(col, pd.Series):
            scores = self.score_sign * col
        else:
            scores = pd.Series(dtype=float)

        scores = scores.dropna()

        if not hasattr(self, "universe") or self.universe is None:
            uni = list(scores.index)
        else:
            uni = [t for t in self.universe if t in scores.index]

        if not uni:
            return pd.Series(dtype=float, name="score")
        scores = scores.loc[uni].astype(float)
        return scores.rename("score")
