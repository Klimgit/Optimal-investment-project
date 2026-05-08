"""Gradient Boosting Regressor baseline for tabular momentum features.

Используем `HistGradientBoostingRegressor` из sklearn как сильный
не-линейный baseline для cross-sectional таргета `target_reg`.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.base import BaseModel


class GBRTModel(BaseModel):
    """Обертка над HistGradientBoostingRegressor."""

    is_classifier = False

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        max_iter: int = 300,
        min_samples_leaf: int = 50,
        random_state: int = 0,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            loss="squared_error",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBRTModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
