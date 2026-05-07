"""I/O утилиты: загрузка YAML-конфигов, безопасное чтение/запись parquet."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name: str) -> dict[str, Any]:
    """Загрузить YAML-конфиг с поддержкой ключа `extends: <other.yaml>`.

    Имя — либо абсолютный путь, либо имя файла из `configs/`.
    """
    path = Path(name)
    if not path.is_absolute():
        path = CONFIGS_DIR / name

    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent is not None:
        parent_cfg = load_config(parent)
        cfg = _deep_merge(parent_cfg, cfg)
    return cfg


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
