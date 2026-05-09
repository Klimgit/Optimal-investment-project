"""Rule-based (non-ML) модели скоринга для momentum-стратегий.

Follow the Winner  — покупаем прошлых победителей (momentum):
    score = +r12  (z-scored 12-мес доходность, высокий скор → long)

Follow the Loser   — покупаем прошлых проигравших (contrarian/mean-reversion):
    score = -r12  (высокий скор → у кого была низкая доходность → long)

Модели не обучаются: fit() — no-op, predict() возвращает нужный столбец X.
Столбцы X поступают в порядке feature_columns() из src.data.features:
    [r1=0, r3=1, r6=2, r12=3, sigma_ann=4, r3_n=5, r6_n=6, r12_n=7, macd…, sig…]
"""
from __future__ import annotations

import numpy as np

from src.models.base import BaseModel

# r12 — индекс 3 в каноническом порядке MOMENTUM_COLS = ["r1","r3","r6","r12",...]
_R12_IDX = 3


class MomentumRuleModel(BaseModel):
    """Скор = direction * X[:, col_idx] без какого-либо обучения.

    Parameters
    ----------
    direction : +1.0 → Follow the Winner, -1.0 → Follow the Loser.
    col_idx   : индекс признака в X (по умолчанию r12 = 3).
    """

    is_classifier = False

    def __init__(self, direction: float = 1.0, col_idx: int = _R12_IDX) -> None:
        self.direction = direction
        self.col_idx = col_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MomentumRuleModel":
        return self  # правило не требует обучения

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.direction * X[:, self.col_idx]
