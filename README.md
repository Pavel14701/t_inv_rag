# Deterministic Trading Engine

Детерминированная система алгоритмического трейдинга: торговые решения принимает **только проверяемый код** — декларативный **DSL** → сигналы → **планируемый Risk Engine (TZ-11)**. ML-модель лишь оценивает вероятность сделки (`P(win)`), а LLM/RAG — вспомогательный контур генерации стратегий, вынесенный за пределы принятия решений.

**Ключевые принципы:**
- решения о входе/выходе принимает только детерминированный код (DSL-интерпретатор + движок исполнения);
- ML-модель оценивает вероятность — она не принимает решений и не имеет доступа к риск-менеджменту;
- LLM/RAG используется только для генерации/модификации стратегий и анализа результатов — вне контура исполнения;
- Risk Engine реализован в рамках TZ-11: лимиты — данные в `configs/risk.yaml`, движок — их интерпретатор; лимиты недоступны для изменения LLM на любом уровне, включая транспорт (в протоколе очередей TZ-09 физически нет такой команды).

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

**5. `rag/` — прикладной RAG-контур** (`dte-rag`)
- LLM-слой: per-request роутинг провайдеров/моделей (Ollama + OpenAI-compatible).
- ingestion (chunk_markdown, white-list доки), vectorstore (InMemory + Qdrant),
  embeddings (Mock + Ollama bge-m3), retrieval, RAGPipeline с pass@1/pass@N
  метриками (TZ-07). 47 тестов.
- Осталось: pass@1 evaluation на живом LLM, интеграционные тесты Qdrant/Ollama.

**6. `main/` — точка входа** (`dte-main`)
- DI (dishka, 4 контура: backtest/inference/rag/api) с провайдерами
  LLM/Embeddings/Qdrant/model bundle/PostgreSQL (TZ-08).
- FastStream-мост WhiteBridge/LocalBridge: ACL, schema-version tolerance,
  reconnect с backoff, heartbeat-мониторинг (TZ-09).
- REST на aiohttp (ingest/strategies/backtests/signals/rag) + PostgreSQL-сторы
  и Alembic-миграции (TZ-10).

### 🚧 В разработке / специфицировано

- **Risk Engine (TZ-11 ✅)** — config-driven движок (5 правил, strict-валидация,
  params-схемы, RISK_* env), встроен в бэктест: reject-аудит в отчёте;
  live-склейка — после FastStream↔PG (TZ-10).
- **`strategies/`** — ядро готово (формат + валидация + реестр + лейблы,
  25 тестов); осталось: склейка с реестром в REST.
- **`backtest/`** — ядро готово (execution/portfolio/engine/metrics/validation
  + baseline gate, 34 теста); осталось: SIV-прогон, msgspec-отчёты, live.
- **`main/`** — DI/REST/PostgreSQL/FastStream-склейка готовы; осталось:
  реальный локальный backtest-runner, aiogram-бот, JWT (TZ-08/09/10).
- **TZ-03 волна 2** — TaProvider: 7 групп индикаторов, multi-output,
  батчевый resolve_history.

### Критерии успеха (TZ-00)

Profit Factor > 1.5 out-of-sample с комиссиями · MaxDD ≤ 20% · Sharpe > 1.0 ·
модель ≥ базлайнов (LR/RF/XGBoost) · RAG pass@1 ≥ 70% ≤ 2 repair-итераций.
*Пока не проверяемы на реальных данных: обучение модели + baseline gate
и live-контур (нужны данные, обученный bundle и FastStream↔PG склейка).*

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
| `dte-strategies` | `strategies/` | polars, niquests | нет |
| `dte-ai` | `ai/` | torch, tensorboard, pyyaml | обучение CUDA / инференс DX12 (extra `gpu`) |
| `dte-infer` | `infer/` | polars, yfinance, t-tech; torch — extra `ml` | нет |
| `dte-rag` | `rag/` | llama-index, qdrant-client, sentence-transformers | нет (Ollama — внешний сервис) |
| `dte-backtest` | `backtest/` | numpy, polars | нет |
| `dte-contracts` | `contracts/` | msgspec | нет |
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
uv run --package dte-ta pytest ta/tests        # тесты индикаторов
uv run --package dte-ai pytest ai/tests        # тесты ai
uv run --package dte-rag pytest rag/tests      # тесты LLM-слоя
uv run --package dte-infer pytest infer/tests  # смоук-тесты инференса
uv run --package dte-strategies pytest strategies/tests  # тесты стратегий
uv run pytest -m dsl                          # только тесты dte-dsl (service-маркеры)
uv run --package dte-infer python -m infer.cli --help   # инференс
```

## Технологии

Python ≥ 3.12 · uv (workspaces) · NumPy/Numba · Polars · TA-Lib · PyTorch ·
LlamaIndex + Ollama · Qdrant · PostgreSQL + SQLAlchemy + Alembic · Redis ·
RabbitMQ (FastStream) · aiogram · dishka.

### Бенчмарки индикаторов ta/ (TZ-12, n=100 000)

Numba-ядра против базлайнов; cold = первый вызов (JIT-компиляция), warm = best-of-3,
фиксированный seed. TA-Lib не установлен в основном окружении (n/a).

| module | indicator | impl | ms | speedup |
|---|---|---|---:|---:|
| overlap | sma | numba (cold) | 1.16 | - |
| overlap | sma | numba (warm) | 0.62 | 1.0x |
| overlap | sma | pandas | 1.04 | 1.7x |
| overlap | ema | numba (warm) | 0.86 | 1.0x |
| overlap | ema | pandas | 0.57 | 0.7x |
| momentum | rsi | numba (warm) | 3.09 | 1.0x |
| momentum | rsi | numpy | 3.14 | 1.0x |
| momentum | macd | numba (warm) | 3.16 | 1.0x |
| momentum | macd | numpy | 3.07 | 1.0x |
| volatility | atr | numba (warm) | 2.05 | 1.0x |
| volatility | atr | pandas | 13.09 | 6.4x |
| candle | cdl_engulfing | numba (warm) | 1.18 | 1.0x |
| custom | scrsi | numba (warm) | 4.09 | 1.0x |
| custom | scrsi | numpy | 3.87 | 0.9x |

Воспроизведение: `uv run python -m ta.benchmarks.run --n 100000 --save`.

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
- [x] TZ-02 strategies + единая OHLC
- [x] TZ-03 ta-dsl provider (волна 1: 10 биндингов × 7 групп)
- [x] TZ-04 backtest (execution + portfolio + engine + metrics + baseline gate)
- [x] TZ-11 Risk Engine (config-driven ядро)
- [ ] TZ-09/10 api bridge + white API
- [ ] TZ-09/10 api bridge + white API skeleton → TZ-07 прикладной RAG → TZ-08 DI-склейка
- [ ] TZ-12 бенчмарки индикаторов · TZ-13 CI

