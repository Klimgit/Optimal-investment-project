"""Тесты chrono-val по группам календарных дат (без утечки внутри даты)."""
from __future__ import annotations

import numpy as np

from src.training.trainer import chronological_group_train_val_mask


def test_chrono_val_uses_whole_last_dates():
    """10 дат × 10 строк: val = ровно последние 2 даты (~20% месяцев)."""
    n_dates, per = 10, 10
    g = np.repeat(np.arange(n_dates, dtype=np.int64), per)
    train_m, val_m = chronological_group_train_val_mask(g, val_frac=0.2)
    assert train_m.sum() == 80
    assert val_m.sum() == 20
    assert set(np.unique(g[val_m])) == {8, 9}
    assert set(np.unique(g[train_m])) == set(range(8))


def test_chrono_val_no_row_from_shared_date_in_train_and_val():
    """Одна та же дата не попадает одновременно в train и val."""
    g = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    train_m, val_m = chronological_group_train_val_mask(g, val_frac=0.34)
                                             
    dates_in_train = set(g[train_m])
    dates_in_val = set(g[val_m])
    assert dates_in_train.isdisjoint(dates_in_val)
