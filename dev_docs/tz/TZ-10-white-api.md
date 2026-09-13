# TZ-10. White API: каркас публичного сервиса

> **Статус: 🔨 ядро + PostgreSQL готовы (12 REST/store-тестов + 10 db-тестов).**
> ✅ main/src/api.py: WhiteAPI — CandleStore (идемпотентный ingest по
> (inst_id, ts), дубль не дублирует — тест), JobStore (202 + job_id pattern),
> SignalStore (latest N), ACL-проверка на submit_backtest.
> ✅ main/src/rest.py: REST по карте п.2.2 на aiohttp (без новых зависимостей —
> aiohttp уже идёт с aiogram): POST /ingest/candles, GET /strategies,
> GET /strategies/{id}, POST /backtests (202), GET /backtests/{job_id},
> GET /signals?ticker=, POST /rag/generate (503 без rag-контура).
> Валидация входов msgspec-структурами из contracts/, 400 на битый JSON.
> ✅ волна 2: PostgreSQL-слой — main/src/db.py (SQLAlchemy 2.0 модели candles/
> strategies/backtest_jobs/signals, переносимая схема: JSON в TEXT, sqlite
> для тестов), main/src/pgstores.py (PgCandleStore/PgJobStore/PgSignalStore —
> те же интерфейсы, что in-memory; WhiteAPI принимает любую реализацию),
> Alembic: migrations/env.py (DATABASE_URL > alembic.ini, psycopg-драйвер) +
> 0001_initial (4 таблицы + 2 индекса; upgrade/downgrade проверены тестом
> на sqlite). DI: DatabaseProvider/DatabasePort в api-контуре (без DATABASE_URL
> → sessions=None → in-memory fallback).
> ✅ волна 2 (склейка): main/src/service.py — make_pg_white_api (WhiteAPI над
> тремя PG-сторами), candle_saver (md.ohlcv → PG, asyncio.to_thread), 
> make_local_bridge/make_white_bridge (cmd.backtest → инъекция runner'а, 
> дефолт — stub с failed-отчётом, никогда не молчит); e2e-тесты над
> TestRabbitBroker: ohlcv → PG идемпотентно, cmd → evt.report → PG job completed.
> ⬜ Реальный локальный runner (DSL→сигналы→backtest+risk) для cmd.backtest.
> ⬜ aiogram-бот поверх тех же контрактов; JWT-аутентификация.
> ⬜ Живой PostgreSQL в CI (service-контейнер) для pgstores.

## 1. Контекст

`main/src` — сейчас скрипт к T-Invest API. Нужен каркас публичного контура: ingest биржевых
данных + REST для веба и aiogram-бота. Не host'ит GPU/ML — только транспорт и хранение.

## 2. Почему именно так

### 2.1. FastStream как основа
FastStream (с RabbitMQ) уже в зависимостях — он же используется в TZ-09; ingest-хендлеры
очередей и publish идут через один фреймворк. **Отвергнутые альтернативы:** чистый aio-pika
(boilerplate), самописный asyncio-цикл (нет retry/ack/сериализации).

### 2.2. Endpoint-карта (v1)
```
POST /ingest/candles        # приём данных бирж (или internal: consume md.* напрямую)
GET  /strategies            # реестр стратегий (read-only)
GET  /strategies/{id}       # + DSL-текст, AST (JSON), метрики
POST /backtests             # запрос бэктеста → cmd.backtest, 202 + job_id
GET  /backtests/{job_id}    # статус/отчёт из evt.report
GET  /signals?ticker=...    # последние сигналы + P(win)
POST /rag/generate          # RAG-генерация DSL (валидация внутри rag-контура)
POST /telegram/webhook      # aiogram
```
**Почему бэктест асинхронный (202 + job):** он исполняется на локали через очередь;
синхронный HTTP на минуты работы — анти-паттерн.

### 2.3. Чего нет в API и почему
- Нет эндпоинтов изменения риск-лимитов, позиций, исполнения ордеров — их нет в протоколе
  очередей (TZ-09 п.2.3), значит их не может быть и здесь.
- Нет прямого доступа к Qdrant/Ollama — только через RAG-контур локали.
- ML-обучение запускается только командой cmd.train (ACL), статусы — через evt.report.

### 2.4. Хранение
PostgreSQL (уже в compose, Alembic настроен): стратегии, отчёты бэктестов, сигналы, jobs.
Рыночные данные — тоже в PG (timescale-совместимая схема позже; сначала простая таблица
candles с уникальным индексом (ticker, ts) — идемпотентность ingest).

### 2.5. Аутентификация
Веб/бот — JWT против white API; сервисный токен — для ingest от биржевых коннекторов.

## 3. Требования

1. Каркас FastStream-приложения: consumers md.* → запись в PG, publishers cmd.*.
2. REST по карте п.2.2 (FastAPI поверх, если потребуется; сначала FastStream + минимальный HTTP).
3. aiogram-бот: те же данные через те же контракты (не отдельная логика).
4. Миграции Alembic для таблиц strategies/backtest_jobs/signals/candles.
5. Валидация всех входов msgspec-структурами из contracts/ (TZ-08).

## 4. Критерии приёмки

- Ingest: идемпотентная запись свечей (дубль не дублирует), retry-safe.
- POST /backtests → cmd.backtest → (мок локали) → evt.report → GET возвращает отчёт.
- Схемы ответов полностью из contracts/, без локальных дубликатов.