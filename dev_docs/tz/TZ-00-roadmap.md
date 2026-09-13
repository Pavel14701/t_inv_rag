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
| 0 | TZ-14 quality baseline | ✅ (ruff единый линтер, корневая pytest-конфигурация + service-маркеры, EN-only guard, mypy clean по всему репо, 0 warnings; полный сюит 2324 passed / 120 skipped) | mypy/линтеры/тесты/языковая дисциплина (EN-only: Numba молча деградирует на не-ASCII) — без безопасной базы рефакторинг TZ-02+ не проверяем |
| 1 | TZ-01 dsl hardening | ✅ (DslValidationError, resolve_history, манифест-маршрутизация в коде) | Все контракты (исключения, провайдеры) строятся на DSL; чинить после появления клиентов дороже |
| 2 | TZ-06 ai stabilization | ✅ (bundle, predict_p_win, YAML, device; остаток — батчеризация OB, замер < 5 мс на реальном железе) | torch в зависимостях, утечка валидации, model bundle — до любого использования ai |
| 3 | TZ-02 strategies + единая OHLC | ✅ (25 тестов: единая схема + Strategy + валидация + реестр + AST + лейбл-генератор) | Формат стратегии и схема данных — склейка ta/dsl/ai; конфликт схем блокирует всё дальше |
| 4 | TZ-03 ta-dsl provider | ✅ (волна 2: универсальный маппер `ta/src/registry.py` — авто-биндинги из сигнатур `*_ind`, 84 индикатора в манифесте/DSL; multi-output с NAMED_OUTPUTS, батчевый resolve, cache-key по всем params — фикс коллизий; волна 1: 4 golden-якоря; 1975 тестов; SMOKE_SKIP: ott = numba-dispatch) | Прокидывание индикаторов в DSL; нужен формат данных из TZ-02 |
| 5 | TZ-04 backtest | ✅ ядро (34 + 12 risk-интеграционных теста; risk-gate и reject-аудит ✅; осталось: SIV-прогон, msgspec-контракты отчётов, live-контур) | Честный бэктест ДО RAG и ДО ML-инференса на реальных данных |
| 6 | TZ-05 inference | ✅ (CLI, конвейер, --ml; остался ручной прогон на T-Invest) | Скрипт сигналов — «бэктест на живом хвосте»; зависит от TZ-02/03/04 (формат стратегии — заготовка) |
| 7 | TZ-09 api bridge | 🔨 транспорт готов (контракты + ACL + WhiteBridge/LocalBridge, reconnect/backoff, heartbeat, e2e через TestRabbitBroker; осталось: живой RabbitMQ, TLS/токены, lag-метрики в Prometheus) | Транспорт white API ↔ локаль; нужны форматы отчётов (TZ-04) и сигналов (TZ-05) |
| 8 | TZ-10 white api skeleton | 🔨 (REST + WhiteAPI, PostgreSQL-слой: модели + PgStores + Alembic + DI DatabaseProvider; FastStream↔PG склейка service.py с e2e; осталось: реальный локальный backtest-runner (DSL→сигналы→backtest+risk), aiogram, JWT, живой PG в CI) | Каркас ingest + REST; после контрактов очередей (TZ-09) |
| 9 | TZ-07 rag | 🔨 ядро готово (LLM-слой, ingestion, vectorstore/embeddings/retrieval, pipeline + pass@1/pass@N метрики, 47 тестов; осталось: pass@1 eval-скрипт над query-set, rag_integration на живом Qdrant/Ollama) | RAG поверх готового формата стратегий и валидатора DSL |
| 10 | TZ-08 contracts/DI | ✅ (DI собрана: 4 контура + rag/Ollama/Qdrant/bundle/DB-провайдеры с Protocol-ключами, 16 тестов; PostgreSQL provider закрыт) | Финальная склейка; фактически ведётся параллельно с TZ-02 |
| 11 | TZ-11 risk engine | ✅ (config-driven движок: 5 правил v1 + params-схемы, strict-валидация, env-оверрайды; интеграция в backtest: reject-аудит в отчёте; 22 + 12 тестов; осталось: live-склейка через TZ-10) | Ключевая фича детерминизма; после TZ-04 (движок исполнения) |
| 12 | TZ-12 ta benchmarks | 🔨 runner готов (7 сценариев × pandas/numpy/TA-Lib базлайны, cold/warm JIT, --save, таблица в README; осталось: воспроизводимость ±10%, TA-Lib/pandas_ta опциональным job в CI, issue на scrsi 0.9x) | Публичное доказательство производительности Numba-ядер |
| 13 | TZ-13 ci | ✅ (GitHub Actions: lint + format + mypy strict + EN-only + 9 pytest matrix jobs — добавлен main) |

> Отступление от порядка: TZ-05 выполнен до TZ-02/03/04 (смок на синтетике допустим —
> формат стратегии в infer остаётся заготовкой до TZ-02).

## 3.1. Свода на текущий момент

- Полный сюит: **2425 passed / 120 skipped / 0 warnings**; ruff и mypy чисты
  по всему репозиторию (единственный принятый tech-debt — докстринги в
  `ta/src/overlap/mama.py`).
- Детерминированный контур готов end-to-end на синтетике:
  ta → DSL → сигналы → **Risk Engine (TZ-11 ✅)** → бэктест с reject-аудитом;
  DSL-манифест расширен до **84 индикаторов** (TZ-03 волна 2 ✅: универсальный
  маппер `ta/src/registry.py`, авто-биндинги из сигнатур `*_ind`).
- Инфраструктура: DI (4 контура), FastStream-мост с ACL, REST, PostgreSQL
  + Alembic, склейка white/local через `service.py` — всё с e2e-тестами
  на in-memory брокере и SQLite; прод-бэкенды (RabbitMQ, PG, Qdrant,
  Ollama) пока не подключались живьём.

## 3.2. Дальнейший порядок работ

1. **Локальный backtest-runner в main** (замыкает TZ-04 live + TZ-09/TZ-10):
   cmd.backtest → DSL→сигналы → `run_backtest(risk_config=...)` → evt.report;
   (TZ-03 волна 2 ✅ — маппер 84 индикаторов уже в DSL), параллельно
   msgspec-контракты отчётов TZ-04.
2. **TZ-07 финал** — pass@1 eval-скрипт (нужен живой Ollama), маркер
   `rag_integration` для живых Qdrant/Ollama (docker compose есть).
3. **TZ-10 добивка** — aiogram-бот, JWT, живой PostgreSQL в CI
   (service-контейнер).
4. **TZ-09 добивка** — живой RabbitMQ, TLS/токены, lag-метрики.
5. **TZ-04/TZ-06 на реальных данных** — SIV-прогон, обучение модели,
   baseline gate (открывает live), батчеризация OB, замер < 5 мс.
6. **TZ-12 финал** — повторные прогоны ±10%, TA-Lib опциональным job в CI.

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