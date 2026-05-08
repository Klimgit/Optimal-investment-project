"""Общий torch-trainer для регрессоров (MLP, LSTM, MC-Dropout).

Особенности под наш use-case:
- мини-данные (60 месяцев × ~1500 акций) → 90k samples, легко влезает в RAM,
  поэтому полный mini-batch training через `TensorDataset` + `DataLoader`;
- Smooth-L1 (Huber) loss — устойчив к жирным хвостам месячных доходностей;
- val-split: последний `val_frac` хронологически из train-окна;
- early stopping по val_loss с patience;
- возврат обученной модели + train/val curves для логирования.

Не спецаллоцирован под классификацию (для MC-Dropout сделаем отдельный
trainer в Фазе 6 со своими loss/predict-конвенциями).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Гиперпараметры тренировки одной torch-модели."""
    epochs: int = 50
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.2
    patience: int = 5
    device: str = "cpu"
    loss: str = "smooth_l1"  # "smooth_l1" | "mse"
    seed: int = 0
    verbose: bool = False


@dataclass
class TrainResult:
    """Результаты тренировки: лучший state_dict + кривые."""
    best_val_loss: float
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    epochs_run: int = 0


def _make_loss(name: str) -> nn.Module:
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    msg = f"Unknown loss: {name}"
    raise ValueError(msg)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_torch_regressor(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
) -> TrainResult:
    """Обучить регрессор с early stopping и вернуть `model` с best weights inplace.

    `X` может быть 2D (для MLP) или 3D (для LSTM, shape `[N, T, F]`).
    Val-split — последние `val_frac` примеров (предполагаем, что
    вызывающий уже отсортировал их хронологически или нет — для нашего
    cross-section это не критично).
    """
    _seed_all(config.seed)
    device = torch.device(config.device)
    model = model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    n = X_t.shape[0]
    n_val = max(int(n * config.val_frac), 1) if config.val_frac > 0 else 0
    n_train = n - n_val

    if n_train < 16:
        msg = f"Not enough training data: n_train={n_train}"
        raise ValueError(msg)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(config.seed))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr = X_t[train_idx], y_t[train_idx]
    X_va, y_va = X_t[val_idx], y_t[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=config.batch_size, shuffle=True, drop_last=False,
    )

    loss_fn = _make_loss(config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    res = TrainResult(best_val_loss=best_val)

    X_va_dev = X_va.to(device)
    y_va_dev = y_va.to(device)

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_dev)
            val_loss = loss_fn(val_pred, y_va_dev).item() if n_val > 0 else float("nan")

        res.train_losses.append(train_loss)
        res.val_losses.append(val_loss)
        res.epochs_run = epoch + 1

        if config.verbose:
            logger.info("epoch %d  train=%.5f  val=%.5f", epoch, train_loss, val_loss)

        if n_val > 0 and val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > config.patience:
                if config.verbose:
                    logger.info("Early stop at epoch %d (best val=%.5f)", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    res.best_val_loss = best_val
    return res


def predict_torch(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 1024) -> np.ndarray:
    """Прогнать `model` на `X`, вернуть 1D numpy-массив скоров."""
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    out = []
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            xb = X_t[i:i + batch_size].to(dev)
            out.append(model(xb).cpu().numpy())
    pred = np.concatenate(out, axis=0)
    return pred.reshape(-1)


def predict_torch_mc(
    model: nn.Module,
    X: np.ndarray,
    n_samples: int = 30,
    device: str = "cpu",
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """MC-Dropout инференс: K forward-passes с включённым dropout.

    Returns
    -------
    mean : `[N]` усреднённый logit (или raw output) по K сэмплам.
    std  : `[N]` стандартное отклонение по K сэмплам — мера эпистемической
           неопределённости: высокая std → модель «не уверена».

    Примечание: оставляем `model.train()`, чтобы Dropout-слои оставались
    активны во время инференса. Все остальные `nn.Dropout`-вызовы должны
    быть включены — это и есть суть Monte-Carlo Dropout.
    """
    model.train()  # keep dropout layers active
    dev = torch.device(device)
    model = model.to(dev)
    X_t = torch.tensor(X, dtype=torch.float32)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = []
            for i in range(0, X_t.shape[0], batch_size):
                xb = X_t[i:i + batch_size].to(dev)
                out.append(model(xb).cpu())
            samples.append(torch.cat(out, dim=0))
    stack = torch.stack(samples, dim=0)  # [K, N, 1]
    mean = stack.mean(dim=0).numpy().reshape(-1)
    std = stack.std(dim=0).numpy().reshape(-1)
    return mean, std
