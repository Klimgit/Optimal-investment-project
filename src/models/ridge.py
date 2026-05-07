"""Ridge-регрессор для скоринга momentum-факторов (бейзлайн)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.models.base import BaseModel


class RidgeModel(BaseModel):
    """Тонкая обёртка над `sklearn.linear_model.Ridge`.

    Parameters
    ----------
    alpha : сила L2-регуляризации.
    """

    is_classifier = False

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
