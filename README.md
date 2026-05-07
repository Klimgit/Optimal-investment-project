# DL momentum strategy

DL-стратегия momentum-инвестирования (MLP, LSTM, MC-Dropout) с бейзлайнами Ridge и LogReg+L2.
Бэктестинг через `quant-pml` на бесплатном Kaggle-датасете US-акций (~1980–2017),
сравнение с S&P 500 по Sharpe / IR / MaxDD / α.

## Quick start

`quant-pml` требует ровно Python 3.12.5. Самый простой способ получить такую версию — через [uv](https://github.com/astral-sh/uv).

```bash
# 0. Установить uv (один раз)
brew install uv

# 1. Окружение на ровно Python 3.12.5
uv python install 3.12.5
uv venv .venv --python 3.12.5
source .venv/bin/activate
uv pip install -e ".[dev]"     # подтянет в т.ч. quant-pml==0.1.16, torch==2.8.0

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

## Структура

```
configs/        # YAML с гиперпараметрами и параметрами бэктеста
src/
  data/         # loader, universe, features
  models/       # ridge, logreg, mlp, lstm
  training/     # trainer, hp_search
  backtest/     # обёртка под quant-pml, portfolio, hedge
  evaluation/   # metrics, plots
  utils/        # seed, io
notebooks/      # 01_data_eda ... 06_final_comparison
tests/          # юнит-тесты на фичи и портфель
results/        # predictions, weights, pnl, reports
```

См. `Description.md` — исходное ТЗ; план находится в `.cursor/plans/`.
