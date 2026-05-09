import numpy as np
import pandas as pd

from src.evaluation.block_bootstrap import annualized_sharpe_daily, block_bootstrap_metric_ci


def test_annualized_sharpe_daily():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 500)
    s = annualized_sharpe_daily(r)
    assert np.isfinite(s)


def test_block_bootstrap_deterministic_seed():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0.0002, 0.008, len(idx)), index=idx)
    a = block_bootstrap_metric_ci(s, n_bootstrap=100, block_len=15, random_state=1)
    b = block_bootstrap_metric_ci(s, n_bootstrap=100, block_len=15, random_state=1)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
