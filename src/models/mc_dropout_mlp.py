"""MC-Dropout MLP-классификатор для предсказания top-decile (`target_clf`).

Идея: dropout-as-Bayesian-approximation (Gal & Ghahramani, 2016).
Во время инференса делаем K forward-passes со включённым dropout — получаем
распределение предсказаний. Среднее → сам скор (proba), std → эпистемическая
неопределённость, по которой можно фильтровать «неуверенные» позиции.

Архитектура:

    input(F) → Linear(h1) → ReLU → Dropout(p) → Linear(h2) → ReLU → Dropout(p) → Linear(1)

p = 0.5 (выше, чем у обычного MLP — для MC-Dropout важна сильная регуляризация).
Loss = BCEWithLogitsLoss.
Inference = `predict()` → mean proba; `predict_with_uncertainty()` → (proba, std logit).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.base import BaseModel
from src.training.trainer import (
    TrainConfig,
    predict_torch_mc,
    train_torch_regressor,
)


class _MCDropoutNet(nn.Module):
    """MLP для классификации с явными Dropout-слоями (включаются в MC-инференсе)."""

    def __init__(self, in_dim: int, hidden: tuple[int, int] = (64, 32), dropout: float = 0.5) -> None:
        super().__init__()
        h1, h2 = hidden
        self.fc1 = nn.Linear(in_dim, h1)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h1, h2)
        self.do2 = nn.Dropout(dropout)
        self.head = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.do1(x)
        x = torch.relu(self.fc2(x))
        x = self.do2(x)
        return self.head(x)              


class MCDropoutMLPClassifier(BaseModel):
    """MC-Dropout MLP для предсказания вероятности топ-децильной доходности."""

    is_classifier = True

    def __init__(
        self,
        hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.5,
        n_mc_samples: int = 30,
        epochs: int = 50,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        val_frac: float = 0.2,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.n_mc_samples = n_mc_samples
        self.train_cfg = TrainConfig(
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, val_frac=val_frac,
            seed=seed, device=device, loss="bce",
        )
        self.device = device
        self._model: _MCDropoutNet | None = None
        self._train_result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MCDropoutMLPClassifier":
        if X.ndim != 2:
            msg = f"MCDropoutMLPClassifier expects 2D X, got shape {X.shape}"
            raise ValueError(msg)
        in_dim = X.shape[1]
        self._model = _MCDropoutNet(in_dim=in_dim, hidden=self.hidden, dropout=self.dropout)
        y_float = np.asarray(y).astype(np.float32)
        if len(np.unique(y_float)) < 2:
            self._fallback_proba = float(y_float.mean()) if len(y_float) > 0 else 0.5
            self._model = None
            return self
        self._train_result = train_torch_regressor(self._model, X, y_float, self.train_cfg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Среднее предсказание K сэмплов, преобразованное в вероятность."""
        if self._model is None:
            return np.full(X.shape[0], getattr(self, "_fallback_proba", 0.5), dtype=float)
        mean_logit, _ = predict_torch_mc(
            self._model, X, n_samples=self.n_mc_samples,
            device=self.device, batch_size=4096,
        )
        return _sigmoid(mean_logit)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns: (proba_mean, proba_std).

        std считаем по logit-сэмплам, потом конвертим в proba-пространство
        через sigmoid'(mean) * std_logit (delta-method approx) — этого
        достаточно для ранжирования по неопределённости.
        """
        if self._model is None:
            n = X.shape[0]
            return np.full(n, getattr(self, "_fallback_proba", 0.5)), np.zeros(n)
        mean_logit, std_logit = predict_torch_mc(
            self._model, X, n_samples=self.n_mc_samples,
            device=self.device, batch_size=4096,
        )
        proba = _sigmoid(mean_logit)
        proba_std = proba * (1 - proba) * std_logit
        return proba, proba_std


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
