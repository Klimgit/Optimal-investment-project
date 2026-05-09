from pathlib import Path

import pandas as pd

from src.evaluation.run_registry import discover_run_directories, load_run


def test_load_run_agg_tmp(tmp_path: Path):
    d = tmp_path / "foo_agg"
    d.mkdir()
    pd.DataFrame({"mean": [0.1]}).to_csv(d / "agg_metrics.csv")
    pd.DataFrame({"mean_total_r": [0.001]}, index=pd.bdate_range("2010-01-01", periods=5)).to_parquet(
        d / "mean_returns.parquet"
    )
    r = load_run(d)
    assert r is not None and r["name"] == "foo" and r["kind"] == "agg"


def test_discover_skips_seed_when_agg_exists(tmp_path: Path):
    (tmp_path / "mlp_agg").mkdir()
    pd.DataFrame({"mean": [0.0]}).to_csv(tmp_path / "mlp_agg" / "agg_metrics.csv")
    pd.DataFrame({"x": [0.0]}, index=pd.bdate_range("2010-01-01", periods=3)).to_parquet(
        tmp_path / "mlp_agg" / "mean_returns.parquet"
    )
    (tmp_path / "mlp_s0").mkdir()
    pd.DataFrame({"value": [1.0]}, index=["sharpe"]).to_csv(tmp_path / "mlp_s0" / "metrics.csv")
    pd.DataFrame({"x": [0.0]}, index=pd.bdate_range("2010-01-01", periods=3)).to_parquet(
        tmp_path / "mlp_s0" / "total_returns.parquet"
    )
    pairs = discover_run_directories(tmp_path, exclude=(), include_seeds=False)
    names = [r[1]["name"] for r in pairs]
    assert "mlp" in names
    assert len([n for n in names if n == "mlp"]) == 1
