# 📘 t_inv_rag — Система алгоритмического трейдинга

Модульная система алгоритмического трейдинга: декларативный **DSL** для торговых условий, высокопроизводительный слой **индикаторов** (NumPy/Numba/Polars), **Transformer-модель** для оценки вероятности сделки и жёсткий детерминированный **риск-менеджмент**.

**Ключевые принципы:**
- логика сигналов описывается декларативным DSL, без исполняемого кода;
- решения о входе/выходе принимает только детерминированный код;
- ML-модель лишь оценивает вероятность — она не принимает решений и не имеет доступа к риск-менеджменту;
- LLM/RAG используется только для генерации/модификации стратегий и анализа результатов;
- риск-лимиты зашиты в код и недоступны для изменения LLM.

---

## Состояние проекта

### ✅ Реализовано

**1. `ta/` — библиотека технических индикаторов**
- NumPy + Numba ядра, Polars-совместимость, без pandas в вычислениях.
- Группы индикаторов:
  - `overlap/` — SMA, EMA, WMA, RMA, HMA, KAMA, ALMA, JMA, TEMA, VIDYA, SuperTrend, Ichimoku и др.;
  - `momentum/` — RSI, MACD, Stoch, CCI, TSI, ROC и др.;
  - `volatility/` — ATR, True Range, BBands, AccBands;
  - `trend/` — ADX, RWI, ZigZag; `statistics/` — stdev, zscore, entropy и др.;
  - `volume/` — VWMA; `candle/` — полный набор паттернов (CDL) + Heikin Ashi, Renko, Kagi;
  - `custom/` — OTT, SCRSI, RSI Clouds, AVS-семейство, market structure.
- Покрыто тестами (momentum, overlap, volatility и др.).

**2. `dsl/` — DSL для торговых условий**
- Токенизатор → парсер → AST → интерпретатор (Visitor), синхронная и асинхронная оценка.
- Поддержка: арифметика, сравнения, логика с short-circuit, вызовы индикаторов с параметрами и атрибутами, исторические смещения `close[1]`, `rising`/`falling`, `let`-биндинги.
- Провайдеры данных: in-process, HTTP, манифесты индикаторов (строгая валидация).
- Подробная документация в `dsl/README.md` и `dsl/docs/`.
- Полный набор тестов (tokenizer, parser, interpreter, providers, integration).

**3. `ai/` — Transformer-модель (Entry-Exit Transformer)**
- Multi-head Transformer: предсказание действия (hold/entry/exit), исхода (win/loss, R-multiple) и паттернов по окну свечей и order blocks.
- Supervised и semi-supervised self-training (псевдо-разметка только для высокоуверенных предсказаний).
- Датасет на скользящих окнах, валидационные контракты (формы, dtypes, конечные значения), метрики (win rate, profit factor), TensorBoard.
- Документация в `ai/docs/` (architecture, model, data, training, quickstart).

**4. Инфраструктура**
- `docker-compose.yaml`: PostgreSQL 18, Redis 8, RabbitMQ 4.2, Qdrant (GPU), Ollama (DeepSeek-R1 8B, GPU).
- `alembic.ini` + `migrations/` — настроены миграции БД.

### 🚧 В разработке / каркас

- **`main/`** — точка входа сервиса и конфигурация (каркас).
- **`strategies/`** — application- и infrastructure-слои (каркас: типы, адаптер индикаторов).
- **RAG-контур** (Qdrant + LlamaIndex + Ollama) — подключение инфраструктуры готово, прикладной слой в разработке.
- **Risk Engine** — специфицирован (см. исходное ТЗ), реализация впереди.

---

## Архитектура

```
[Данные] → [Индикаторы (ta)] → [DSL → AST → сигналы] → [ML-модель → вероятность]
                                                        ↓
                                               [Risk Engine → решение]
                                                        ↓
                                                 [Исполнение]
```

LLM/RAG работает **параллельно**, а не внутри контура принятия решений.

---

## Структура репозитория

```
ta/          # индикаторы (NumPy/Numba), тесты
dsl/         # DSL: tokenizer, parser, ast, interpreter, providers, тесты
ai/          # Transformer-модель, датасеты, обучение, метрики, docs
strategies/  # слой стратегий (каркас)
main/        # сервис, конфигурация
migrations/  # Alembic-миграции
dev_docs/    # спецификации: DSL, индикаторы, свечи, статистика, чеклист
docker-compose.yaml, alembic.ini, pyproject.toml
```

## Технологии

Python ≥ 3.12 · NumPy/Numba · Polars · TA-Lib · PyTorch (sentence-transformers, TensorBoard) · LlamaIndex + Ollama · Qdrant · PostgreSQL + SQLAlchemy + Alembic · Redis · RabbitMQ (FastStream) · aiogram · dishka.
Менеджмент зависимостей — **uv** (`uv.lock`).

## Запуск

```bash
uv sync                     # установка зависимостей
docker compose up -d        # инфраструктура (postgres, redis, rabbitmq, qdrant, ollama)
uv run alembic upgrade head # миграции БД
uv run pytest               # тесты (dsl/tests, ta/src/tests)
```

Тесты отдельных пакетов: `uv run pytest dsl/tests` (конфиг `dsl/pytest.ini`), `uv run pytest ta/src/tests`.

## Качество кода

Ruff (+ format), flake8, mypy, isort — конфигурация в `pyproject.toml` и `setup.cfg`. Линия — 79 символов, target Python 3.12.

---

## Ограничения (неизменные принципы)

- LLM не принимает торговых решений.
- LLM не меняет риск-параметры.
- LLM не генерирует код исполнения сделок.
- Все решения принимаются только кодом.

## Roadmap

- [ ] Реализация Risk Engine (лимиты риска/просадки, обязательные SL/TP, фильтры).
- [ ] Прикладной RAG-контур: генерация и модификация стратегий в DSL, анализ бэктестов.
- [ ] Развитие слоя стратегий и основного сервиса (`main/`).
- [ ] Обучение модели на исторических данных и интеграция с сигнальным контуром.
