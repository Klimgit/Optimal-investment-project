"""Обнаружение папок прогонов в ``results/`` (single / agg / опционально *_sN)."""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_SEED_RE = re.compile(r"_s\d+$")


def load_run(run_dir: Path) -> dict | None:
    """Формат ``single-run`` или ``multi-seed agg`` → dict или None."""
    agg_metrics = run_dir / "agg_metrics.csv"
    mean_ret = run_dir / "mean_returns.parquet"
    if agg_metrics.exists() and mean_ret.exists():
        agg = pd.read_csv(agg_metrics, index_col=0)
        metrics = agg["mean"] if "mean" in agg.columns else agg.iloc[:, 0]
        returns = pd.read_parquet(mean_ret)
        name = run_dir.name.removesuffix("_agg")
        return {"name": name, "metrics": metrics, "returns": returns, "kind": "agg"}

    metrics_p = run_dir / "metrics.csv"
    returns_p = run_dir / "total_returns.parquet"
    if metrics_p.exists() and returns_p.exists():
        metrics = pd.read_csv(metrics_p, index_col=0)["value"]
        returns = pd.read_parquet(returns_p)
        return {"name": run_dir.name, "metrics": metrics, "returns": returns, "kind": "single"}
    return None


def discover_run_directories(
    root: Path,
    *,
    exclude: Iterable[str] = ("_summary",),
    only: Iterable[str] | None = None,
    include_seeds: bool = False,
) -> list[tuple[Path, dict]]:
    """Список ``(path, load_run(path))`` для всех валидных прогонов."""
    exclude_set = set(exclude)
    candidates: list[Path] = []
    for p in sorted(root.iterdir()) if root.exists() else []:
        if not p.is_dir() or p.name in exclude_set:
            continue
        candidates.append(p)

    out: list[tuple[Path, dict]] = []
    for p in candidates:
        if not include_seeds and _SEED_RE.search(p.name):
            continue
        run = load_run(p)
        if run is None:
            continue
        out.append((p, run))

    if only:
        allow = set(only)
        out = [(p, r) for p, r in out if r["name"] in allow]
    return out


def resolve_rebal_weights_path(run_dir: Path) -> Path | None:
    """``rebal_weights.parquet`` у прогона или у ``*_s0`` для ``*_agg``."""
    direct = run_dir / "rebal_weights.parquet"
    if direct.exists():
        return direct
    name = run_dir.name
    if name.endswith("_agg"):
        base = name.removesuffix("_agg")
        s0 = run_dir.parent / f"{base}_s0" / "rebal_weights.parquet"
        if s0.exists():
            return s0
    return None
