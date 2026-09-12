# Deterministic Trading Engine

Детерминированная система алгоритмического трейдинга: торговые решения принимает **только проверяемый код** — декларативный **DSL** → сигналы → **планируемый Risk Engine (TZ-11)**. ML-модель лишь оценивает вероятность сделки (`P(win)`), а LLM/RAG — вспомогательный контур генерации стратегий, вынесенный за пределы принятия решений.

**Ключевые принципы:**
- решения о входе/выходе принимает только детерминированный код (DSL-интерпретатор + движок исполнения);
- ML-модель оценивает вероятность — она не принимает решений и не имеет доступа к риск-менеджменту;
- LLM/RAG используется только для генерации/модификации стратегий и анализа результатов — вне контура исполнения;
- Risk Engine и жёсткие риск-лимиты пока **не реализованы**; после реализации в рамках TZ-11 лимиты будут зашиты в код и недоступны для изменения LLM на любом уровне, включая транспорт.

> Историческое имя репозитория `t_inv_rag` отражало первую итерацию (RAG поверх T-Invest).
> Ядро проекта — детерминированный торговый контур; RAG — вспомогательный слой
> (см. раздел «RAG-контур» и TZ-07).

---

## Состояние проекта

### ✅ Реализовано

**1. `ta/` — библиотека технических индикаторов** (`dte-ta`)
- NumPy + Numba ядра (`@njit`), Polars-совместимость, без pandas в вычислениях.
- Группы: `overlap/` (SMA…JMA, SuperTrend, Ichimoku), `momentum/` (RSI, MACD, Stoch…),
  `volatility/` (ATR, BBands), `trend/` (ADX, ZigZag), `statistics/`, `volume/`,
  `candle/` (CDL-паттерны, Heikin Ashi, Renko, Kagi), `custom/` (OTT, SCRSI, AVS…).
- Каузальность и NaN-контракт; **1952 теста** зелёные.

**2. `dsl/` — DSL торговых условий** (`dte-dsl`)
- Токенизатор → парсер → AST → интерпретатор (Visitor), sync + async.
- Арифметика, логика с short-circuit, вызовы индикаторов, смещения `close[1]`,
  `rising/falling`, `let`-биндинги. **Нет `eval`/`exec` — произвольный код неисполним.**
- Провайдеры: in-process, HTTP (HTTP/2/3), манифесты со строгой валидацией;
  маршрутизация по манифесту, `resolve_history`, `DslValidationError` (TZ-01 ✅).
- **154 теста** зелёные.

**3. `ai/` — Entry-Exit Transformer** (`dte-ai`)
- Предсказывает: действие (hold/entry/exit), исход (win/loss + R-multiple), паттерны —
  по окну свечей и order blocks.
- Supervised + self-training (псевдо-разметка), каузальные TP/SL от ATR(t-1),
  хронологический train/val сплит (без утечки валидации), model bundle +
  контракт инференса `predict_p_win` (TZ-06 ✅).
- YAML-конфиг (`configs/ai.yaml`), воспроизводимость (seed), 71 тест.
- Обученных артефактов пока нет — модель не обучена на реальных данных
  (обучение — после TZ-04).

**4. `infer/` — скрипт инференса** (`dte-infer`)
- CLI: свечи (synthetic/parquet/yfinance/tinvest) → DSL-сигналы → `--ml` фильтр по
  `P(win)` (TZ-05 ✅). 9 смоук-тестов.

**5. `rag/llm.py` — LLM-слой** (`dte-rag`)
- Per-request роутинг провайдеров/моделей (Ollama + OpenAI-compatible), TZ-07 п.0 ✅.
- Прикладной RAG-контур (ingestion/retrieval/generation) — не реализован.

### 🚧 В разработке / специфицировано

- **`strategies/`** — каркас; формат `Strategy` + `manifest_hash` + реестр — TZ-02 (следующий шаг).
- **`backtest/`** — единый движок исполнения (лейблы/бэктест/live) — TZ-04.
- **Risk Engine** — специфицирован, реализация — TZ-11.
- **`main/`** — точка входа, DI (dishka), API-бридж — TZ-08/09/10.

### Критерии успеха (TZ-00)

Profit Factor > 1.5 out-of-sample с комиссиями · MaxDD ≤ 20% · Sharpe > 1.0 ·
модель ≥ базлайнов (LR/RF/XGBoost) · RAG pass@1 ≥ 70% ≤ 2 repair-итераций.
*Пока не проверяемы: бэктест (TZ-04) впереди.*

> ⚠️ **Baseline gate** — сравнение Transformer с простыми методами (Buy & Hold,
> логистическая регрессия, RF/XGBoost) на одном тестовом периоде — обязательный гейт:
> без него результаты бэктеста невалидны, обучение на реальных данных и live-контур
> не открываются (TZ-04 п.4.6.1). Не превосходит на > 5–10% по Sharpe/PF → упрощаем
> архитектуру.

---
## Монорепозиторий (uv workspaces)

Каждый сервис — отдельный член workspace со своим окружением и зависимостями.

```bash
uv sync --all-packages        # полное dev-окружение (все члены workspace)
uv sync --package dte-dsl     # изолированное окружение одного пакета
uv run --package dte-dsl pytest dsl/tests
```

| Пакет | Каталог | Ключевые зависимости | GPU |
|-------|---------|----------------------|-----|
| `dte-ta` | `ta/` | numba, numpy, scipy | нет |
| `dte-dsl` | `dsl/` | niquests | нет |
| `dte-strategies` | `strategies/` | polars, pandas, ta-lib | нет |
| `dte-ai` | `ai/` | torch, tensorboard, pyyaml | обучение CUDA / инференс DX12 (extra `gpu`) |
| `dte-infer` | `infer/` | polars, yfinance, t-tech; torch — extra `ml` | нет |
| `dte-rag` | `rag/` | llama-index, qdrant-client, sentence-transformers | нет (Ollama — внешний сервис) |
| `dte-main` | `main/` | dishka, faststream, aiogram, sqlalchemy, alembic | нет |

Принцип TZ-00 п.4.7: **GPU-зависимости (torch) существуют только в `dte-ai`**
(и опционально в `dte-infer[ml]`) — публичный контур их не тянет.

Import-имена пакетов (`ta`, `dsl`, `ai`, `infer`, `rag`, `main`) не изменились —
изменилась упаковка, а код и ~2200 тестов остались совместимыми.

## Архитектура

```
[Данные] → [Индикаторы (ta)] → [DSL → AST → сигналы] → [ML: P(win)] → [Risk Engine (TZ-11, в разработке) → решение] → [Исполнение]
                                детерминированный контур ─────────────────────────────────────┘
[LLM/RAG] → генерация/объяснение стратегий (вне контура решений)
```

LLM/RAG работает **параллельно**, а не внутри контура принятия решений.

## Запуск

```bash
uv sync --all-packages
docker compose up -d          # postgres, redis, rabbitmq, qdrant, ollama
uv run --package dte-main alembic upgrade head   # миграции БД
uv run --package dte-dsl pytest dsl/tests      # тесты DSL
uv run --package dte-ta pytest ta/src/tests    # тесты индикаторов
uv run --package dte-ai pytest ai/src/tests    # тесты ai
uv run --package dte-rag pytest rag/tests      # тесты LLM-слоя
uv run --package dte-infer pytest infer/tests  # смоук-тесты инференса
uv run --package dte-infer python -m infer.cli --help   # инференс
```

## Технологии

Python ≥ 3.12 · uv (workspaces) · NumPy/Numba · Polars · TA-Lib · PyTorch ·
LlamaIndex + Ollama · Qdrant · PostgreSQL + SQLAlchemy + Alembic · Redis ·
RabbitMQ (FastStream) · aiogram · dishka.

## Качество кода

Ruff (+format) — единственный линтер (flake8/isort подлежат ликвидации в TZ-14),
mypy — поэтапное ужесточение (строгий режим сначала ai/infer), линия 79 символов,
Python 3.12. **Код, идентификаторы, докстринги, комментарии — только английский**
(Numba/тулчейн деградируют на не-ASCII в `.py`); русский — только в `*.md`-доках.
Конвенция тестов: `dev_docs/testing_convention.md`. CI — TZ-13.

## Документация

- `dev_docs/tz/` — TZ-00…TZ-13: roadmap, ТЗ модулей, статусы.
- `dsl/README.md`, `dsl/docs/` — DSL; `ai/docs/` — модель и обучение;
  `dev_docs/quant_checklist.md` — сверка с индустриальным чек-листом.

## Ограничения (неизменные принципы)

- LLM не принимает торговых решений.
- LLM не меняет риск-параметры (в протоколе очередей TZ-09 нет такой команды).
- LLM не генерирует код исполнения сделок.
- Все решения принимаются только кодом.

## Roadmap

Актуальный порядок — в `dev_docs/tz/TZ-00-roadmap.md`. Кратко:

- [x] TZ-01 DSL hardening · TZ-06 ai stabilization · TZ-05 inference
- [ ] **TZ-02 strategies + единая OHLC** (следующий шаг)
- [ ] TZ-03 ta-dsl provider → TZ-04 backtest (единый движок исполнения)
- [ ] TZ-11 Risk Engine
- [ ] TZ-09/10 api bridge + white API skeleton → TZ-07 прикладной RAG → TZ-08 DI-склейка
- [ ] TZ-12 бенчмарки индикаторов · TZ-13 CI

