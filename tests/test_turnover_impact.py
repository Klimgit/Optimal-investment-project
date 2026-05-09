import numpy as np
import pandas as pd

from src.evaluation.market_impact import approximate_daily_turnover_fraction, linear_impact_haircut


def test_turnover_spread_and_haircut():
    idx = pd.bdate_range("2010-01-01", periods=10)
                                            
    r = pd.DataFrame(
        [[0.5, -0.5], [0.4, -0.4]],
        index=[idx[0], idx[5]],
        columns=["A", "B"],
    )
    daily = pd.bdate_range("2010-01-01", periods=15)
    t = approximate_daily_turnover_fraction(r, daily)
    assert (t >= 0).all()
    assert t.sum() > 0
    ret = pd.Series(np.full(len(daily), 0.001), index=daily)
    adj = linear_impact_haircut(ret, t, eta=0.01)
    assert (adj <= ret).all()
