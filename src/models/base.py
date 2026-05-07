"""Общий интерфейс ML-модели для DL-momentum-стратегии.

Стратегия (`MLScoringStrategy`) дёргает у модели только две операции:
- `fit(X, y)`     — обучение на (X, y) train-окна;
- `predict(X)`    — скоринг на тех же фичах для inference.

Регрессоры предсказывают доходность, классификаторы — вероятность top-decile.
Конкретная реализация (Ridge, LogReg, MLP, LSTM, MC-Dropout) живёт в `ridge.py`,
`logreg.py`, `mlp.py`, `lstm.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Минимальный sklearn-подобный интерфейс."""

    is_classifier: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Скаляр-скор на актив (regression: ŷ; classification: P(top decile))."""
        ...
