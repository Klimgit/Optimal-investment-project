import numpy as np
import pandas as pd

from src.evaluation.block_bootstrap import block_bootstrap_paired_ci
from src.evaluation.holdout import collect_holdout_metrics, paired_daily_metrics_numpy, slice_period


def test_slice_period():
    idx = pd.bdate_range("2015-01-01", periods=400)
    s = pd.Series(np.linspace(0, 1, len(idx)), index=idx)
    (out,) = slice_period(s, start="2016-01-01")
    assert out.index.min() >= pd.Timestamp("2016-01-01")


def test_collect_holdout_metrics_smoke():
    idx = pd.bdate_range("2015-01-01", periods=200)
    rng = np.random.default_rng(0)
    strat = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0002, 0.009, len(idx)), index=idx)
    m = collect_holdout_metrics(strat, bench, eval_start="2015-07-01")
    assert m["n_days"] > 0
    assert "sharpe_ann_daily_xs" in m


def test_paired_bootstrap_deterministic():
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(1)
    strat = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0002, 0.009, len(idx)), index=idx)
    a = block_bootstrap_paired_ci(strat, bench, n_bootstrap=80, block_len=15, random_state=3)
    b = block_bootstrap_paired_ci(strat, bench, n_bootstrap=80, block_len=15, random_state=3)
    assert a["sharpe_ann_daily_xs"]["ci_low"] == b["sharpe_ann_daily_xs"]["ci_low"]


def test_paired_numpy_matches_series():
    idx = pd.bdate_range("2016-01-01", periods=100)
    rng = np.random.default_rng(2)
    st = rng.normal(0.0004, 0.008, len(idx))
    bt = rng.normal(0.0003, 0.008, len(idx))
    s = pd.Series(st, index=idx)
    b = pd.Series(bt, index=idx)
    m1 = paired_daily_metrics_numpy(st, bt, rf=0.0)
    m2 = collect_holdout_metrics(s, b, eval_start=None, eval_end=None)
    assert np.isclose(m1["sharpe_ann_daily_xs"], m2["sharpe_ann_daily_xs"], rtol=1e-5)
