# DL momentum strategy

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

Скрипты сохраняют артефакты в `results/{strategy}/`; для multi-seed запусков также создаются
`results/{strategy}_s{seed}/` и агрегаты `results/{strategy}_agg/`. Логирование — в `mlruns/`.

```bash
# 5a. Бейзлайны: Ridge (regression) и LogReg+L2 (classification), полный OOS 2010-2017.
PYTHONPATH=. python scripts/run_baselines.py
PYTHONPATH=. python scripts/run_baselines.py --strategies ridge gbrt logreg follow_winner follow_loser

# 5b. DL-регрессоры: MLP и LSTM, по 5 сидов каждый.
PYTHONPATH=. python scripts/run_dl_reg.py --strategies mlp lstm --seeds 5

# 5c. MC-Dropout MLP-классификатор, с/без uncertainty-filter, по 5 сидов.
PYTHONPATH=. python scripts/run_dl_clf.py --strategies mc_dropout mc_dropout_filtered --seeds 5

# 6. Сводный отчёт по всем стратегиям (по умолчанию multi-seed агрегаты).
PYTHONPATH=. python scripts/compare_runs.py             # один ряд на стратегию
PYTHONPATH=. python scripts/compare_runs.py --main-chart-all  # comparison.png со всеми найденными стратегиями
PYTHONPATH=. python scripts/compare_runs.py --include-seeds  # все per-seed runs
PYTHONPATH=. python scripts/compare_runs.py --markdown results/_summary/report.md

# 7. UI MLflow со сравнением run-ов, графиками, метриками.
mlflow ui --backend-store-uri ./mlruns
```

Финальный сравнительный отчёт удобнее смотреть в `notebooks/04_strategies_comparison.ipynb`.

Исследовательские выводы и бэклог — **`docs/research_roadmap.md`**. Литература и аудит реализации — **`docs/literature_review.md`**. Протокол holdout — **`docs/holdout_protocol.md`**.

```bash
PYTHONPATH=. python scripts/holdout_metrics.py results/mlp_best_agg --write
PYTHONPATH=. python scripts/bootstrap_metrics.py results/mlp_best_agg --paired --eval-start 2016-01-01 --metrics all --json
PYTHONPATH=. python scripts/compare_holdout_quality.py --impact-eta 0.00015   # таблица по всем runs → results/_summary/holdout_quality.csv
```

## Как добавить non-ML стратегию

Минимальная схема для rule-based стратегии без обучения:

1. Создайте модель-скорер в `src/models/rule_based.py` (или в новом файле `src/models/`) с интерфейсом `BaseModel`:
   - `fit(...)` — no-op, если обучение не нужно;
   - `predict(X)` — возвращает `np.ndarray` скорингов (чем выше скор, тем выше шанс попасть в long).

2. Подключите стратегию в `scripts/run_baselines.py` в функции `_strategy_specs()`:
   - добавьте имя стратегии,
   - `model_factory`,
   - `target_col` (обычно `target_reg`),
   - `model_params` (для логов и MLflow).

3. Добавьте имя в `choices` аргумента `--strategies` в `run_baselines.py`.

4. Запустите стратегию:
   ```bash
   PYTHONPATH=. python scripts/run_baselines.py --strategies <new_strategy_name>
   ```

5. Добавьте её в сравнительный график:
   ```bash
   PYTHONPATH=. python scripts/compare_runs.py --main-chart-order follow_winner follow_loser <new_strategy_name> lstm mlp_best bench_low_vol_ls --no-low-vol-line
   ```

6. Проверьте артефакты:
   - `results/<new_strategy_name>/metrics.csv`
   - `results/_summary/metrics.csv`
   - `results/_summary/comparison.png`
