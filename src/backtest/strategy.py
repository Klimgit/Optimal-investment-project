"""Стратегия для скоринга активов произвольной ML-моделью под `quant_pml`.

Контракт `BaseMLScoringStrategy` (см. quant_pml/strategies/ml/scoring/...):
- `_fit(training_data)`            — обучаем модель на train-окне;
- `predict_scores(prediction_data)` → `pd.Series` со скорами на pred_date.

Сами фичи берём из нашего `data/features/panel.parquet`, индексированного по
`(date, ticker)`. quant_pml не «знает» про этот файл — это наша внутренняя
кухня. От рантайма используем только `pred_date` и `strategy.universe`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_pml.strategies.ml.scoring.base_ml_scoring_strategy import BaseMLScoringStrategy
from quant_pml.strategies.optimization_data import PredictionData, TrainingData

from src.models.base import BaseModel


@dataclass
class TrainSlice:
    """Срез фич/таргета для тренировки на одном train-окне.

    Если `sequence_length=1`, `X` имеет форму `[N, F]` — обычная табличка.
    Если `sequence_length>1`, `X` имеет форму `[N, T, F]` — последовательности
    из T месячных снимков, нужные LSTM/RNN-моделям.
    """
    X: np.ndarray
    y: np.ndarray
    feature_cols: list[str]
    n_snapshots: int
    n_rows: int


class MLScoringStrategy(BaseMLScoringStrategy):
    """Универсальная стратегия: модель обучается на нашем panel и скорит активы.

    Parameters
    ----------
    model_factory : фабрика, возвращающая свежий `BaseModel` на каждый ребаланс
        (для `retrain=from_scratch`).
    panel_path : путь до `data/features/panel.parquet`.
    feature_cols : список столбцов-фич (24 для нашего setup); если None —
        все колонки кроме `date, ticker, ret_next, target_reg, target_clf`.
    target_col : `target_reg` (regression) или `target_clf` (classification).
    train_window_months : сколько месяцев истории берём в train.
    mode : `long_short` / `long_only` / `short_only` (передаётся в quant_pml).
    quantile : доля верхнего/нижнего квантиля (0.1 = top/bot 10%).
    weighting_scheme : `equally_weighted` (наш дефолт).
    """

    def __init__(
        self,
        model_factory: Callable[[], BaseModel],
        panel_path: str | Path = "data/features/panel.parquet",
        feature_cols: list[str] | None = None,
        target_col: str = "target_reg",
        train_window_months: int = 60,
        sequence_length: int = 1,
        mode: str = "long_short",
        quantile: float | None = 0.1,
        n_holdings: int | None = None,
        weighting_scheme: str = "equally_weighted",
    ) -> None:
        super().__init__(
            mode=mode,
            quantile=quantile,
            n_holdings=n_holdings,
            weighting_scheme=weighting_scheme,
        )
        self.model_factory = model_factory
        self.panel_path = Path(panel_path)
        self.target_col = target_col
        self.train_window_months = train_window_months
        self.sequence_length = sequence_length

        self._panel: pd.DataFrame | None = None
        self._feature_cols: list[str] | None = feature_cols
        self._fitted_model: BaseModel | None = None

    def _load_panel(self) -> pd.DataFrame:
        if self._panel is None:
            df = pd.read_parquet(self.panel_path)
            df["ticker"] = df["ticker"].astype(str)
            df["date"] = pd.to_datetime(df["date"])
            self._panel = df.set_index(["date", "ticker"]).sort_index()
            if self._feature_cols is None:
                exclude = {"ret_next", "target_reg", "target_clf"}
                self._feature_cols = [c for c in self._panel.columns if c not in exclude]
        return self._panel

    def _slice_train(self, pred_date: pd.Timestamp) -> TrainSlice:
        """Срезать panel на train-окне ≤ pred_date (исключая саму pred_date)."""
        panel = self._load_panel()
        feat_cols = self._feature_cols  # type: ignore[assignment]
        train_start = pred_date - pd.DateOffset(months=self.train_window_months)

        snap_dates = panel.index.get_level_values("date")
        mask = (snap_dates >= train_start) & (snap_dates < pred_date)
        train = panel.loc[mask].dropna(subset=feat_cols + [self.target_col])

        if self.sequence_length <= 1:
            X = train[feat_cols].to_numpy(dtype=np.float64)
            y = train[self.target_col].to_numpy(dtype=np.float64)
            return TrainSlice(
                X=X, y=y, feature_cols=feat_cols,
                n_snapshots=int(snap_dates[mask].nunique()), n_rows=int(len(train)),
            )

        X, y = self._build_sequences_from_panel(panel, train, feat_cols)
        return TrainSlice(
            X=X, y=y, feature_cols=feat_cols,
            n_snapshots=int(snap_dates[mask].nunique()), n_rows=int(X.shape[0]),
        )

    def _build_sequences_from_panel(
        self,
        panel: pd.DataFrame,
        anchor_rows: pd.DataFrame,
        feat_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Собрать `[N, T, F]` последовательности для каждого (date, ticker)
        из `anchor_rows`, беря историю `T = sequence_length` месяцев из `panel`.

        Если для какого-то якоря не хватает истории — пропускаем.
        """
        T = self.sequence_length
        feat_arr = panel[feat_cols].to_numpy(dtype=np.float32)
        target_arr = panel[self.target_col].to_numpy(dtype=np.float32)

        # Position lookup: (date, ticker) → row index in `panel`.
        pos_lookup: dict[tuple[pd.Timestamp, str], int] = {}
        for i, idx in enumerate(panel.index):
            pos_lookup[idx] = i

        sorted_dates = panel.index.get_level_values("date").unique().sort_values()
        date_pos = {d: j for j, d in enumerate(sorted_dates)}

        Xs: list[np.ndarray] = []
        ys: list[float] = []
        for (anchor_date, ticker), _row in anchor_rows.iterrows():
            j = date_pos.get(anchor_date)
            if j is None or j + 1 < T:
                continue
            window_dates = sorted_dates[j + 1 - T: j + 1]
            seq_rows = []
            ok = True
            for d in window_dates:
                ridx = pos_lookup.get((d, ticker))
                if ridx is None:
                    ok = False
                    break
                row = feat_arr[ridx]
                if not np.all(np.isfinite(row)):
                    ok = False
                    break
                seq_rows.append(row)
            if not ok:
                continue
            seq = np.stack(seq_rows, axis=0)  # [T, F]
            Xs.append(seq)
            target_idx = pos_lookup[(anchor_date, ticker)]
            t_val = target_arr[target_idx]
            if not np.isfinite(t_val):
                Xs.pop()
                continue
            ys.append(float(t_val))

        if not Xs:
            return np.empty((0, T, len(feat_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.float32)

    def _fit(self, training_data: TrainingData) -> None:
        pred_date = pd.Timestamp(training_data.pred_date)
        slc = self._slice_train(pred_date)

        if slc.n_rows < 100:
            self._fitted_model = None
            return

        model = self.model_factory()
        model.fit(slc.X, slc.y)
        self._fitted_model = model

    def predict_scores(self, prediction_data: PredictionData) -> pd.Series:
        """Скор на каждый тикер universe для prediction_data.pred_date."""
        if self._fitted_model is None:
            return pd.Series(dtype=float, name="score")

        panel = self._load_panel()
        pred_date = pd.Timestamp(prediction_data.pred_date)

        snap_dates = panel.index.get_level_values("date").unique().sort_values()
        valid = snap_dates[snap_dates <= pred_date]
        if len(valid) == 0:
            return pd.Series(dtype=float, name="score")
        snap_date = valid.max()

        snap = panel.loc[snap_date]
        feat_cols = self._feature_cols  # type: ignore[assignment]
        snap = snap.dropna(subset=feat_cols)

        if not hasattr(self, "universe") or self.universe is None:
            uni = list(snap.index)
        else:
            uni = [t for t in self.universe if t in snap.index]

        if not uni:
            return pd.Series(dtype=float, name="score")

        snap = snap.loc[uni]

        if self.sequence_length <= 1:
            X = snap[feat_cols].to_numpy(dtype=np.float64)
            scores = self._fitted_model.predict(X)
            return pd.Series(scores, index=snap.index, name="score")

        T = self.sequence_length
        idx_pos = {d: j for j, d in enumerate(snap_dates)}
        j = idx_pos[snap_date]
        if j + 1 < T:
            return pd.Series(dtype=float, name="score")
        window_dates = snap_dates[j + 1 - T: j + 1]

        feat_arr = panel[feat_cols].to_numpy(dtype=np.float32)
        pos_lookup: dict[tuple[pd.Timestamp, str], int] = {}
        for i, idx in enumerate(panel.index):
            pos_lookup[idx] = i

        Xs: list[np.ndarray] = []
        kept: list[str] = []
        for ticker in uni:
            seq_rows = []
            ok = True
            for d in window_dates:
                ridx = pos_lookup.get((d, ticker))
                if ridx is None:
                    ok = False
                    break
                row = feat_arr[ridx]
                if not np.all(np.isfinite(row)):
                    ok = False
                    break
                seq_rows.append(row)
            if not ok:
                continue
            Xs.append(np.stack(seq_rows, axis=0))
            kept.append(ticker)
        if not Xs:
            return pd.Series(dtype=float, name="score")
        X = np.stack(Xs, axis=0)
        scores = self._fitted_model.predict(X)
        return pd.Series(scores, index=kept, name="score")
