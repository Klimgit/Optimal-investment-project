"""Воспроизводимость: фиксируем все источники случайности."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Зафиксировать seed для random/numpy/torch и переменных окружения.

    `deterministic_torch=True` включает детерминированные kernel'ы CuDNN ценой
    некоторого замедления — для учебного проекта это приемлемо и важнее
    воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
