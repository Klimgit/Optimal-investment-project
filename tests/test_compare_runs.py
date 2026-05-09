"""Smoke-тест для `scripts/compare_runs.py` (через прямой импорт `_load_run`)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

                                               
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_runs                              


def _make_single_run(d: Path, *, seed: int = 0) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=200)
    pd.DataFrame({"total_r": rng.normal(0, 0.01, 200)}, index=idx).to_parquet(d / "total_returns.parquet")
    pd.Series({"sharpe": 0.5, "final_nav": 1.05, "max_dd": -0.1}, name="value").to_csv(d / "metrics.csv", header=True)


def _make_agg_run(d: Path, *, seed: int = 1) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=200)
    pd.DataFrame({"mean_total_r": rng.normal(0, 0.01, 200)}, index=idx).to_parquet(d / "mean_returns.parquet")
    df = pd.DataFrame(
        {"mean": [0.6, 1.1, -0.08], "std": [0.1, 0.05, 0.02]},
        index=["sharpe", "final_nav", "max_dd"],
    )
    df.to_csv(d / "agg_metrics.csv")


def test_load_run_single(tmp_path):
    d = tmp_path / "ridge"
    _make_single_run(d)
    run = compare_runs._load_run(d)
    assert run is not None
    assert run["kind"] == "single"
    assert run["name"] == "ridge"
    assert "sharpe" in run["metrics"].index


def test_load_run_agg(tmp_path):
    d = tmp_path / "mlp_agg"
    _make_agg_run(d)
    run = compare_runs._load_run(d)
    assert run is not None
    assert run["kind"] == "agg"
    assert run["name"] == "mlp"                       
    assert float(run["metrics"]["sharpe"]) == 0.6


def test_load_run_neither(tmp_path):
    d = tmp_path / "garbage"
    d.mkdir()
    (d / "random.txt").write_text("not a run")
    assert compare_runs._load_run(d) is None
