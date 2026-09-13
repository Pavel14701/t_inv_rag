# TZ-00. Roadmap: сведение модулей в единую систему

## 1. Назначение

Корневой документ. Каждое ТЗ (TZ-01..TZ-13) самодостаточно, порядок выполнения определён здесь.
Проект: **Deterministic Trading Engine** (бывш. `t_inv_rag`) — детерминированный торговый
контур; RAG/LLM — вспомогательный слой вне контура решений.

## 1.1. Монорепозиторий (uv workspaces)

Каждый модуль — член uv workspace с изолированным окружением:

| Пакет | Каталог | Ключевые зависимости | Тесты |
|-------|---------|----------------------|-------|
| `dte-ta` | `ta/` | numba, numpy, scipy | `uv run --package dte-ta pytest ta/tests` |
| `dte-dsl` | `dsl/` | niquests | `uv run --package dte-dsl pytest dsl/tests` |
| `dte-strategies` | `strategies/` | polars | `uv run --package dte-strategies pytest strategies/tests` (TZ-02) |
| `dte-ai` | `ai/` | torch, tensorboard, pyyaml | `uv run --package dte-ai pytest ai/tests` |
| `dte-infer` | `infer/` | polars, yfinance, t-tech (+`ml`: torch) | `uv run --package dte-infer pytest infer/tests` |
| `dte-rag` | `rag/` | llama-index, qdrant, sentence-transformers | `uv run --package dte-rag pytest rag/tests` |
| `dte-risk` | `risk/` | numpy, pyyaml | `uv run --package dte-risk pytest risk/tests` |
| `dte-main` | `main/` | dishka, faststream, aiogram, sqlalchemy, alembic | — (TZ-08/10) |

Принципы монорепозитория:
1. **Изоляция окружений**: `uv sync --package dte-<x>` поднимает окружение только с
   зависимостями этого пакета; полный dev-env — `uv sync --all-packages`.
2. **GPU-зависимости изолированы** (п.4.7): torch существует только в `dte-ai`
   (и опционально `dte-infer[ml]`); `dte-dsl`/`dte-ta`/`dte-strategies` его не тянут.
3. **Import-имена неизменны** (`ta`, `dsl`, `ai`, `infer`, `rag`, `main`): упаковка
   меняется, код и существующие тесты не ломаются. Физический src-layout не вводится
   до появления вторых потребителей (решение зафиксировано здесь, revisit — TZ-08).
   Следствие: члены workspace — виртуальные пакеты, поэтому кросс-пакетные
   зависимости (например, `dte-infer` → niquests/numba/scipy для dsl/ta) объявляются
   в pyproject члена напрямую, а не как workspace-зависимости.
4. Каждый член workspace имеет собственный `pyproject.toml` (+ `pytest.ini`),
   корневой проект — только dev-инструменты и общий тестовый стек.

## 2. Карта системы

```
Биржи (T-Invest, OKX) ──► [White API: ingest + REST + aiogram]   (публичный контур, без GPU)
                                  │  RabbitMQ over TLS
                                  │  соединение инициирует ЛОКАЛЬНЫЙ узел (outbound,
                                  ▼  входящие порты на локали закрыты)
                          [Локальный GPU-узел: ai-обучение/инференс,
                           ta+DSL движок, backtest, RAG, Qdrant, Ollama]
```

Внутри локального узла:

```
ta ──(TaProvider, TZ-03)──► dsl ──► backtest (TZ-04) ──► отчёты ──► White API
                         ▲                    ▲
              strategies (TZ-02)          ai P(win) (TZ-06)
```

## 3. Порядок выполнения и статусы

Легенда: ✅ выполнено · 🔨 в работе · ⬜ не начато (порядок зафиксирован TZ-00).

| # | ТЗ | Статус | Почему именно здесь |
|---|----|--------|---------------------|
| 0 | TZ-14 quality baseline | ⬜ стартовая точка | mypy/линтеры/тесты/языковая дисциплина (EN-only: Numba молча деградирует на не-ASCII) — без безопасной базы рефакторинг TZ-02+ не проверяем |
| 1 | TZ-01 dsl hardening | ✅ (DslValidationError, resolve_history, манифест-маршрутизация в коде) | Все контракты (исключения, провайдеры) строятся на DSL; чинить после появления клиентов дороже |
| 2 | TZ-06 ai stabilization | ✅ (bundle, predict_p_win, YAML, device; остаток — батчеризация OB, замер < 5 мс) | torch в зависимостях, утечка валидации, model bundle — до любого использования ai |
| 3 | TZ-02 strategies + единая OHLC | ✅ (25 тестов: единая схема + Strategy + валидация + реестр + AST + лейбл-генератор) | Формат стратегии и схема данных — склейка ta/dsl/ai; конфликт схем блокирует всё дальше |
| 4 | TZ-03 ta-dsl provider | 🔨 (волна 1: 4 индикатора + TaProvider + 11 тестов; осталось: 7 групп, multi-output, resolve_history) | Прокидывание индикаторов в DSL; нужен формат данных из TZ-02 |
| 5 | TZ-04 backtest | ✅ (34 теста: execution + portfolio + engine + metrics + validation + baseline gate) | Честный бэктест ДО RAG и ДО ML-инференса на реальных данных |
| 6 | TZ-05 inference | ✅ (CLI, конвейер, --ml; остался ручной прогон на T-Invest) | Скрипт сигналов — «бэктест на живом хвосте»; зависит от TZ-02/03/04 (формат стратегии — заготовка) |
| 7 | TZ-09 api bridge | 🔨 (транспорт готов: контракты + ACL + FastStream-мост WhiteBridge/LocalBridge, reconnect/backoff, heartbeat; осталось: живой RabbitMQ, TLS) | Транспорт white API ↔ локаль; нужны форматы отчётов (TZ-04) и сигналов (TZ-05) |
| 8 | TZ-10 white api skeleton | 🔨 (WhiteAPI stores + REST на aiohttp по карте эндпоинтов, 7 тестов; осталось: PostgreSQL/Alembic, FastStream↔PG склейка, aiogram) | Каркас ingest + REST; после контрактов очередей (TZ-09) |
| 9 | TZ-07 rag | 🔨 (ядро RAG готово: LLM-слой, ingestion, vectorstore/embeddings/retrieval, pipeline + pass@1/pass@N метрики, 47 тестов; осталось: pass@1 eval-скрипт, живой Qdrant/Ollama) | RAG поверх готового формата стратегий и валидатора DSL |
| 10 | TZ-08 contracts/DI | 🔨 (DI собрана: 4 контура + rag/Ollama/Qdrant/bundle-провайдеры с Protocol-ключами, 15 тестов; осталось: PostgreSQL provider) | Финальная склейка; фактически ведётся параллельно с TZ-02 |
| 11 | TZ-11 risk engine | 🔨 (спека config-driven; скелет dte-risk: конфиг+реестр правил+check()+22 теста зелёные) | Ключевая фича детерминизма; после TZ-04 (движок исполнения) |
| 12 | TZ-12 ta benchmarks | 🔨 (runner ta/benchmarks/run.py: 7 сценариев × pandas/numpy/TA-Lib базлайны, cold/warm JIT, --save в results/*.md; осталось: README-раздел, воспроизводимость ±10%) | Публичное доказательство производительности Numba-ядер |
| 13 | TZ-13 ci | ✅ (GitHub Actions: lint + mypy + EN-only + 8 pytest matrix) |

> Отступление от порядка: TZ-05 выполнен до TZ-02/03/04 (смок на синтетике допустим —
> формат стратегии в infer остаётся заготовкой до TZ-02).

## 4. Сквозные принципы (обязательны для всех ТЗ)

1. **LLM вне контура принятия решений.** LLM генерирует/объясняет стратегии; вход/выход
   решает детерминированный код. Риск-лимиты недоступны LLM на любом уровне, включая
   транспорт: в протоколе очередей TZ-09 физически нет команды «изменить лимиты».
2. **Один движок исполнения на три потребителя** (лейблы ai, бэктест, live) — TZ-04 п.0.
   Три реализации = расхождение семантики сделок между P(win), бэктестом и реальностью.
3. **Единая OHLC-схема**: `open, high, low, close, volume` (TZ-02 п.3).
4. **Look-ahead-инвариант**: функция на баре t получает только `[:t+1]`; проверяется тестом
   подмены «будущих» баров мусором (TZ-03 п.5, TZ-04 п.5).
5. **NaN-контракт**: warm-up = NotReady (сигнал False + счётчик пропусков), NaN после
   прогрева = ошибка данных (TZ-01 п.5, TZ-03 п.4).
6. **Воспроизводимость**: seed (numpy/torch/random) + конфиг + git hash в каждом отчёте.
7. **GPU-зависимости только на локальном узле** (TZ-09): requirements публичного контура
   не содержат torch.

## 5. Критерии общего успеха (из quant_checklist.md)

> ⚠️ **Baseline gate — главный фильтр.** Сравнение модели с простыми методами
> (Buy & Hold, логрегрессия, RF/XGBoost) — обязательный гейт: без пройденного
> сравнения результаты бэктеста не считаются валидными, обучение на реальных
> данных и live-контур не открываются. Детали — TZ-04 п.4.6.1.

- Profit Factor > 1.5 на out-of-sample с комиссиями; MaxDD ≤ 20%; Sharpe > 1.0.
- **Gate:** модель превосходит базлайны (логистическая регрессия, Random Forest/XGBoost,
  Buy & Hold) на **> 5–10% по Sharpe или Profit Factor** по совокупным метрикам
  (Accuracy, F1, Profit Factor, Sharpe) — иначе упрощение архитектуры / возврат к простой модели.
- Повторный запуск с тем же seed даёт идентичный отчёт.
- RAG: ≥ 70% запросов дают валидный DSL ≤ 2 repair-итераций (pass@1).