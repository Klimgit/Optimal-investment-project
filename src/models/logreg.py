"""Logistic Regression + L2 для классификационного скоринга (target_clf).

Используется как бейзлайн против MLP-классификатора (с MC-Dropout) в Фазе 6.
В качестве "score" возвращаем `predict_proba(X)[:, 1]` — вероятность класса 1
(top decile). Сортировка top/bot 10% потом сделает ровно ту же операцию,
что и для регрессии.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseModel


class LogRegL2Model(BaseModel):
    """Тонкая обёртка над `sklearn.linear_model.LogisticRegression` (L2).

    Parameters
    ----------
    C : обратная сила регуляризации (`1/lambda`). Меньше C => сильнее L2.
    max_iter : предел итераций L-BFGS.
    """

    is_classifier = True

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self._model = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegL2Model":
        y_int = np.asarray(y).astype(int)
        if len(np.unique(y_int)) < 2:
            self._model = None
            self._fallback_proba = float(y_int.mean()) if len(y_int) > 0 else 0.5
            return self
        self._model.fit(X, y_int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.full(X.shape[0], self._fallback_proba, dtype=float)
        return self._model.predict_proba(X)[:, 1]
