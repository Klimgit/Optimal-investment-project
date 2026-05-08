"""Тесты `load_benchmark_returns`."""
from __future__ import annotations

import pandas as pd

from src.evaluation.benchmarks import load_benchmark_returns


def test_load_from_exact_folder(tmp_path):
    d = tmp_path / "ridge"
    d.mkdir()
    idx = pd.bdate_range("2015-01-01", periods=5)
    pd.DataFrame({"b": [0.01, -0.01, 0.02, 0.0, 0.01]}, index=idx).to_parquet(d / "benchmark_returns.parquet")
    s = load_benchmark_returns(tmp_path, "ridge")
    assert s is not None
    assert len(s) == 5


def test_load_from_seed_folder(tmp_path):
    d = tmp_path / "mlp_s2"
    d.mkdir(parents=True)
    idx = pd.bdate_range("2015-01-01", periods=3)
    pd.DataFrame({"b": [0.0, 0.0, 0.01]}, index=idx).to_parquet(d / "benchmark_returns.parquet")
    s = load_benchmark_returns(tmp_path, "mlp")
    assert s is not None
    assert len(s) == 3


def test_missing_returns_none(tmp_path):
    assert load_benchmark_returns(tmp_path, "nope") is None
