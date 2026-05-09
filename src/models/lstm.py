"""LSTM-регрессор для скоринга акций по месячным sequence-фичам.

Вход: `[N, T, F]` — N акций × T месячных снимков × F фичей.
Архитектура:

    LSTM(F → hidden, num_layers=L, dropout=p) → last hidden state →
    Linear(hidden → 1)

Стратегия `MLScoringStrategy(sequence_length=12)` собирает T=12 предыдущих
ребаланс-снимков для каждой пары (snap_date, ticker) и пропускает через LSTM.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from src.models.base import BaseModel
from src.training.trainer import TrainConfig, predict_torch, train_torch_regressor


class _LSTMNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                      
        out, _ = self.lstm(x)
        last = out[:, -1, :]               
        return self.head(last)


class LSTMRegressor(BaseModel):
    """LSTM-регрессор для скоринга акций (`target_reg`)."""

    is_classifier = False

    def __init__(
        self,
        hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        epochs: int = 30,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 4,
        val_frac: float = 0.2,
        val_split_mode: str = "random",
        grad_clip: float | None = None,
        scheduler: str | None = None,
        cosine_eta_min: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.train_cfg = TrainConfig(
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, val_frac=val_frac,
            val_split_mode=val_split_mode, grad_clip=grad_clip, scheduler=scheduler,
            cosine_eta_min=cosine_eta_min,
            seed=seed, device=device, loss="smooth_l1",
        )
        self.device = device
        self._model: _LSTMNet | None = None
        self._train_result = None

    def fit(self, X: np.ndarray, y: np.ndarray, val_group_ids: np.ndarray | None = None) -> "LSTMRegressor":
        if X.ndim != 3:
            msg = f"LSTMRegressor expects 3D X [N,T,F], got shape {X.shape}"
            raise ValueError(msg)
        in_dim = X.shape[2]
        self._model = _LSTMNet(
            in_dim=in_dim, hidden=self.hidden, num_layers=self.num_layers,
            dropout=self.dropout, bidirectional=self.bidirectional,
        )
        cfg = replace(self.train_cfg, val_group_ids=val_group_ids)
        self._train_result = train_torch_regressor(self._model, X, y, cfg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            msg = "Model not fitted yet"
            raise RuntimeError(msg)
        return predict_torch(self._model, X, device=self.device, batch_size=2048)
