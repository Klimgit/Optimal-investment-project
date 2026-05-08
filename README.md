# DL momentum strategy

DL-стратегия momentum-инвестирования (MLP, LSTM, MC-Dropout) с бейзлайнами Ridge и LogReg+L2.
Бэктестинг через [`quant-pml`](https://pypi.org/project/quant-pml/) на бесплатном Kaggle-датасете
US-акций (~1980–2017), сравнение с S&P 500 по Sharpe / IR / MaxDD / α.
Логирование экспериментов через MLflow (local file store, без серверов).

## Quick start

`quant-pml` требует ровно Python 3.12.5. Простейший способ — через [uv](https://github.com/astral-sh/uv).

```bash
# 0. Установить uv (один раз)
brew install uv

# 1. Окружение на ровно Python 3.12.5
uv python install 3.12.5
uv venv .venv --python 3.12.5
source .venv/bin/activate
uv pip install -e ".[dev]"     # подтянет quant-pml, torch, mlflow и т.д.

# 2. Данные
#    Kaggle CLI должен быть авторизован (~/.kaggle/kaggle.json, chmod 600).
bash scripts/download_kaggle.sh                # ~700 MB, в data/raw/kaggle/
python scripts/download_spx.py                 # бенчмарк ^GSPC -> data/processed/spx.parquet

# 3. Препроцессинг
python -m src.data.loader                      # *.us.txt -> data/processed/prices.parquet
python -m src.data.universe                    # Top-1500/мес -> data/processed/universe.parquet
python -m src.data.features                    # 24 фичи + target -> data/features/panel.parquet

# 4. Тесты
PYTHONPATH=. pytest -q
```

## Запуск бэктестов

Все скрипты сохраняют артефакты в `results/{strategy}/` и логируют в `mlruns/`.

```bash
# 5a. Бейзлайны: Ridge (regression) и LogReg+L2 (classification), полный OOS 2010-2017.
PYTHONPATH=. python scripts/run_baselines.py

# 5b. DL-регрессоры: MLP и LSTM, по 5 сидов каждый.
PYTHONPATH=. python scripts/run_dl_reg.py --strategies mlp lstm --seeds 5

# 5c. MC-Dropout MLP-классификатор, с/без uncertainty-filter, по 5 сидов.
PYTHONPATH=. python scripts/run_dl_clf.py --strategies mc_dropout mc_dropout_filtered --seeds 5

# 6. Сводный отчёт по всем стратегиям (по умолчанию multi-seed агрегаты).
PYTHONPATH=. python scripts/compare_runs.py             # один ряд на стратегию
PYTHONPATH=. python scripts/compare_runs.py --include-seeds  # все per-seed runs
PYTHONPATH=. python scripts/compare_runs.py --markdown results/_summary/report.md

# 7. UI MLflow со сравнением run-ов, графиками, метриками.
mlflow ui --backend-store-uri ./mlruns
```

Финальный сравнительный отчёт удобнее смотреть в `notebooks/04_strategies_comparison.ipynb`.

## Структура проекта

```
configs/                  # YAML с дефолтами (configs/base.yaml, mlp_*.yaml, lstm_reg.yaml)
src/
  data/                   # loader.py, universe.py, features.py
  models/                 # base, ridge, logreg, mlp, lstm, mc_dropout_mlp
  training/               # trainer.py (torch + early stop + MC-Dropout инференс)
  backtest/               # experiment_config, dataset_adapter, strategy, mc_strategy
  evaluation/             # plots, artifacts, tracker (MLflow), benchmarks.py
  utils/                  # seed, io
notebooks/                # 01_data_eda, 04_strategies_comparison
tests/                    # юнит-тесты, синтетика; PYTHONPATH=. pytest -q
scripts/                  # download_*, smoke_backtest, run_*, compare_runs
results/                  # артефакты бэктестов (per-strategy + _agg + _summary)
mlruns/                   # MLflow file-store
```

## Архитектурные решения

- **Стратегия = адаптер под `quant_pml.BaseMLScoringStrategy`**:
  `_fit(training_data)` обучает модель на нашем `data/features/panel.parquet`,
  обрезанном по `pred_date - train_window_months`; `predict_scores(prediction_data)`
  возвращает `pd.Series` со скорами на ребаланс-дате. Top/bot 10% L/S и
  dollar-neutral сортировку делает родительский класс quant-pml.

- **DatasetData собирается одним адаптером** (`src/backtest/dataset_adapter.py`):
  long-prices → wide pivot, добавление колонок `rf` (нули) и `spx-rf` (excess SPX),
  daily presence_matrix из месячного universe. Никаких внутренних данных
  `quant_pml` (CRSP/Compustat) не используем.

- **Multi-seed для DL**: каждый seed = независимый full-OOS backtest. Усреднение
  кривых post-hoc (`scripts/run_dl_reg.py` → `*_agg/mean_returns.parquet`).

- **MC-Dropout**: dropout остаётся активен на инференсе (Gal & Ghahramani 2016),
  K=30 сэмплов. Эпистемическая σ может фильтровать «неуверенные» позиции
  (`MCDropoutScoringStrategy`).

См. `Description.md` — исходное ТЗ.
