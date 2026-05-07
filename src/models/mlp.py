"""MLP-регрессор с двумя скрытыми слоями для скоринга momentum-факторов.

Архитектура:

    input(F) → Linear(h1) → ReLU → Dropout(p) → Linear(h2) → ReLU → Dropout(p) → Linear(1)

Loss = Smooth-L1 (Huber): устойчив к выбросам месячных доходностей.
Optimizer = AdamW с weight_decay (это и есть L2-регуляризация на веса).
Early stopping по val_loss с patience.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.base import BaseModel
from src.training.trainer import TrainConfig, predict_torch, train_torch_regressor


class _MLPNet(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, int] = (64, 32), dropout: float = 0.3) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor(BaseModel):
    """MLP-регрессор для скоринга акций (`target_reg`)."""

    is_classifier = False

    def __init__(
        self,
        hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.3,
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
        self.train_cfg = TrainConfig(
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, val_frac=val_frac,
            seed=seed, device=device, loss="smooth_l1",
        )
        self.device = device
        self._model: _MLPNet | None = None
        self._in_dim: int | None = None
        self._train_result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        if X.ndim != 2:
            msg = f"MLPRegressor expects 2D X, got shape {X.shape}"
            raise ValueError(msg)
        self._in_dim = X.shape[1]
        self._model = _MLPNet(in_dim=self._in_dim, hidden=self.hidden, dropout=self.dropout)
        self._train_result = train_torch_regressor(self._model, X, y, self.train_cfg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            msg = "Model not fitted yet"
            raise RuntimeError(msg)
        return predict_torch(self._model, X, device=self.device, batch_size=4096)
