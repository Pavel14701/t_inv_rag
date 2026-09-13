> **Статус: ✅ реализован (ядро: единая схема + Strategy + валидация + реестр + AST).**
> Реализовано: единая OHLC-схема (`open/high/low/close`) с legacy-маппингом
> `*__price` в `PriceDataFramePolars`; `Strategy` + `Metrics` (frozen dataclass);
> `validate_strategy(s, manifest)` = parse + indicator-manifest check;
> `StrategyRegistry` (JSON-файлы в `strategies/data/`, auto-pins manifest_hash);
> `indicators_used(expr)` через рекурсивный walk AST to_dict;
> 25 тестов зелёные. Лейбл-генератор реализован: labels.py -> generate_labels() ->
> action/outcome массивы для ai/src/dataset.py (вход close > open -> сигнал через
> TaProvider -> backtest engine -> action=1/2 + R-multiple). Look-ahead инвариант зелёный.

# TZ-02. Слой стратегий и единая схема данных

## 1. Контекст

`strategies/` — каркас: `PriceDataFramePolars` и функции индикаторов. Нет формата стратегии,
реестра и валидации. Одновременно выявлен конфликт схем данных: `PriceDataFramePolars`
требует `open_price, close_price, high_price, low_price`, а `ai/` и polars-функции `ta/`
работают с `open, high, low, close`. Две канонические схемы OHLC в одном проекте — на
каждом стыке понадобится адаптер, и один из них рано или поздно промахнётся.

## 2. Почему именно так

### 2.1. Единая OHLC-схема: `open, high, low, close, volume`
**Почему эти имена:** (а) `ai/` целиком написан под них (features, quickstart, contract);
(б) `ta`-функции `*_pololars` по умолчанию читают `close_col='close'`; (в) `dev_docs/api.md`
и биржи используют короткие имена. Переименование `PriceDataFramePolars` дешевле, чем
адаптеры во всех потребителях. **Отвергнутая альтернатива** (адаптер-слой): три consumers
× N модулей = O(N) точек ошибки вместо одной.

### 2.2. Формат Strategy с manifest_hash
```python
@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    description: str
    dsl_entry: str            # выражение DSL на вход
    dsl_exit: str | None      # выражение DSL на выход (None = только SL/TP)
    params: dict              # константы для провайдера/фильтров
    manifest_hash: str        # хэш манифеста, с которым валидировалось
    created_at: datetime
    metrics: Metrics | None   # заполняет backtest
```
**Почему manifest_hash обязателен:** стратегия ссылается на индикаторы конкретной версии
манифеста. При добавлении новых индикаторов старые стратегии в RAG-примерах (few-shot)
могут подсунуть LLM несуществующую схему. Payload-фильтр по актуальному хэшу решает это
на этапе retrieval (TZ-07).

### 2.3. Валидация как единственный вход в реестр
`validate_strategy(s, manifest) -> list[str]` = parse обоих выражений + ManifestValidator.
Ничего не сохраняется без прохождения. **Почему:** DSL — машинно-проверяемый артефакт;
пропуск невалидной стратегии ломает сразу бэктест, RAG-примеры и инференс.

### 2.4. Лейбл-генератор живёт здесь
`strategies/src/application/labels.py`: «DSL-сигнал → сделка → outcome» с fill-логикой из
`ai/src/features.py` (вход по open[i+1], комиссия, слиппедж), но поверх движка исполнения
TZ-04 (см. TZ-04 п.0). **Почему:** сейчас лейблы ai/ описывают только order-block-сетапы —
P(win) учится предсказывать исходы чужой торговли. Нужен мост «сигнал стратегии → исход».

## 3. Требования

1. Переименовать колонки `PriceDataFramePolars` в единую схему (п.2.1), обновить consumers.
2. `Strategy` + `validate_strategy` + реестр (JSON-файлы в `strategies/data/`; миграция в
   PostgreSQL через Alembic — позже, когда появится main-сервис).
3. Лейбл-генератор (п.2.4) на движке TZ-04; контракт результата = `action/outcome` массивы,
   совместимые с `ai/src/dataset.py`.
4. Экспорт AST-метаданных: список используемых индикаторов обходом AST (`to_dict` уже есть) —
   для фич ai/ и payload RAG.

## 4. Критерии приёмки

- Эталонная стратегия (`let r = rsi(period=14) in r.value < 30 and rising(close, 5)`)
  проходит validate → бэктест-дым на синтетике → корректные action/outcome массивы.
- Тест-инвариант лейблов: лейбл на баре t не меняется при подмене баров > t (look-ahead).
- `uv run pytest strategies` зелёный.