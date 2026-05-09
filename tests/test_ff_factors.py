"""Тесты переразложения месячных факторов на дневную шкалу."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.ff_factors import monthly_factor_to_daily_simple


def test_monthly_to_daily_reproduces_compound():
    idx_m = pd.date_range("2020-01-31", periods=3, freq="ME")
    monthly = pd.Series([0.01, -0.02, 0.03], index=idx_m, name="SMB")
    di = pd.bdate_range("2020-01-02", "2020-03-31")
    d = monthly_factor_to_daily_simple(monthly, di)
    assert len(d) == len(di)
                                                             
    total = float((1.0 + d).prod() - 1.0)
    manual = float((1.0 + monthly).prod() - 1.0)
    assert np.isclose(total, manual, rtol=1e-9)
