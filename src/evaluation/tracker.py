"""MLflow-обёртка для трекинга экспериментов.

Если `mlflow` не установлен — работаем как no-op, а артефакты всё равно
сохраняются на диск через `artifacts.py` (MLflow вторичен).

Storage by default — local file backend at `mlruns/`. Открыть UI:

    mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    _MLFLOW_AVAILABLE = False
    logger.info("mlflow is not installed — tracker will be a no-op.")


class ExperimentTracker:
    """Очень тонкий фасад над MLflow run-API.

    Использование:

        with ExperimentTracker("dl-momentum").run("ridge_baseline", params=...) as t:
            t.log_metric("sharpe", 1.2)
            t.log_figure(fig, "equity.png")
    """

    def __init__(
        self,
        experiment_name: str = "dl-momentum",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.enabled = enabled and _MLFLOW_AVAILABLE
        self._active = False

        if self.enabled:
            uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)

    @contextmanager
    def run(self, run_name: str, params: dict[str, Any] | None = None):
        if not self.enabled:
            yield self
            return
        with mlflow.start_run(run_name=run_name):
            self._active = True
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            try:
                yield self
            finally:
                self._active = False

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if not (self.enabled and self._active):
            return
        try:
            mlflow.log_metric(key, float(value), step=step)
        except Exception as e:
            logger.warning("Failed to log metric %s: %s", key, e)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not (self.enabled and self._active):
            return
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and pd.notna(v)}
        try:
            mlflow.log_metrics(clean, step=step)
        except Exception as e:
            logger.warning("Failed to log metrics dict: %s", e)

    def log_figure(self, fig: plt.Figure, artifact_name: str) -> None:
        if not (self.enabled and self._active):
            return
        try:
            mlflow.log_figure(fig, artifact_name)
        except Exception as e:
            logger.warning("Failed to log figure %s: %s", artifact_name, e)

    def log_artifact(self, path: str | Path, artifact_subdir: str | None = None) -> None:
        if not (self.enabled and self._active):
            return
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
        except Exception as e:
            logger.warning("Failed to log artifact %s: %s", path, e)
