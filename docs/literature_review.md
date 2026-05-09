### Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*

- **Идея:** классическое эмпирическое описание **momentum** — недавние «победители» по доходности продолжают перевешивать «проигравших» на среднесрочном горизонте.
- **У нас:** правила **follow-the-winner / follow-the-loser** (`scripts/run_baselines.py`), квантильный long-short по скорингам и общая momentum-логика портфеля в духе классических постановок.

---

### Barroso & Santa-Clara (2015), *Momentum Has Its Moments*

- **Идея:** **масштабирование экспозиции под волатильность** снижает просадки и улучшает Sharpe относительно «сырого» momentum.
- **У нас:** линия **Low-risk** на сводных графиках (`results/bench_low_vol_ls`, см. `scripts/run_factor_benchmarks.py`, `scripts/compare_runs.py`) задаётся как отдельный факторный бенчмарк по низкой σ; полное управление риском «как в статье» (динамический таргет vol у основной стратегии) — возможное расширение.

---

### Lim, Zohren & Roberts (2019), *Enhancing Momentum with Deep Learning*

- **Идея:** **LSTM** для оптимизации весов momentum-портфеля (фьючерсный контекст в оригинале), улучшение Sharpe относительно классических постановок.
- **У нас:** **LSTM** и **MLP** в `scripts/run_dl_reg.py`, общий пайплайн с monthly rebalance и обучением на панели признаков (`src/models/lstm.py`, конфиги в `configs/`).

---

### Poh, Lim et al. (2021), *Deep Learning Cross-Sectional Ranking …* / Learning-to-Rank для акций

- **Идея:** вместо точечной **регрессии** доходности — **ранжирование** бумаг в кросс-секции (pairwise / listwise objectives), в работах этого направления отмечается существенный выигрыш по Sharpe относительно MSE-регрессии.
- **У нас:** текущие скоринговые модели в основном **регрессия в обходной постановке** (`target_reg`, `MLScoringStrategy`). Переход на LtR — см. бэклог в `research_roadmap.md`.

---

### Goyal, Welch & Zafirov (2022), *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction*

- **Идея:** систематическая проверка предикторов премии за акции; **большая доля** литературных предикторов **не подтверждается в строгом OOS** или нестабильна.
- **У нас:** напоминание интерпретировать **любые in-sample и даже одиночный OOS-срез** осторожно; имеет смысл опираться на **holdout**, **bootstrap ДИ** (`scripts/bootstrap_metrics.py`), **несколько seeds** для нейросетей и смену периодов / издержек, как в `docs/research_roadmap.md`.

---
