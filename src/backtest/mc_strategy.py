"""MCDropoutScoringStrategy — расширение `MLScoringStrategy` для моделей,
поддерживающих `predict_with_uncertainty(X) -> (mean, std)`.

Идея: после получения скоров отбрасываем «неуверенные» предсказания (где
эпистемическая неопределённость σ выше порога). Остальное — поведение
ровно как у родителя: top/bot 10% по среднему скору.

Порог можно задавать двумя способами:
- `uncertainty_quantile=0.5` — отрезаем верхние 50% по σ (оставляем
  половину самых уверенных). Адаптивно к распределению σ.
- `uncertainty_threshold=0.05` — абсолютный порог σ.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_pml.strategies.optimization_data import PredictionData

from src.backtest.strategy import MLScoringStrategy

logger = logging.getLogger(__name__)


class MCDropoutScoringStrategy(MLScoringStrategy):
    """ML-стратегия с MC-Dropout uncertainty-filter.

    Parameters
    ----------
    uncertainty_quantile : доля наименее уверенных предсказаний, которые
        отбрасываются перед сортировкой top/bot. Если `None` — фильтр
        выключен. Например, `0.5` — оставляем 50% самых уверенных.
    uncertainty_threshold : абсолютный порог σ (если задан вместе с
        quantile — берётся максимум из двух фильтров).
    """

    def __init__(
        self,
        *args,
        uncertainty_quantile: float | None = None,
        uncertainty_threshold: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.uncertainty_quantile = uncertainty_quantile
        self.uncertainty_threshold = uncertainty_threshold

    def predict_scores(self, prediction_data: PredictionData) -> pd.Series:
        if self._fitted_model is None:
            return pd.Series(dtype=float, name="score")

        if not hasattr(self._fitted_model, "predict_with_uncertainty"):
            return super().predict_scores(prediction_data)

        panel = self._load_panel()
        pred_date = pd.Timestamp(prediction_data.pred_date)

        snap_dates = panel.index.get_level_values("date").unique().sort_values()
        valid = snap_dates[snap_dates <= pred_date]
        if len(valid) == 0:
            return pd.Series(dtype=float, name="score")
        snap_date = valid.max()

        snap = panel.loc[snap_date]
        feat_cols = self._feature_cols  # type: ignore[assignment]
        snap = snap.dropna(subset=feat_cols)

        if not hasattr(self, "universe") or self.universe is None:
            uni = list(snap.index)
        else:
            uni = [t for t in self.universe if t in snap.index]
        if not uni:
            return pd.Series(dtype=float, name="score")
        snap = snap.loc[uni]

        X = snap[feat_cols].to_numpy(dtype=np.float64)
        proba, sigma = self._fitted_model.predict_with_uncertainty(X)

        scores = pd.Series(proba, index=snap.index, name="score")
        sigmas = pd.Series(sigma, index=snap.index, name="sigma")

        keep_mask = pd.Series(True, index=scores.index)
        if self.uncertainty_quantile is not None and 0 < self.uncertainty_quantile < 1:
            cutoff = sigmas.quantile(self.uncertainty_quantile)
            keep_mask &= sigmas <= cutoff
        if self.uncertainty_threshold is not None:
            keep_mask &= sigmas <= self.uncertainty_threshold

        scores = scores[keep_mask]
        if scores.empty:
            logger.debug("All scores filtered out at %s; returning unfiltered.", pred_date.date())
            return pd.Series(proba, index=snap.index, name="score")
        return scores
