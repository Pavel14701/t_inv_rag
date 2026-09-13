# TZ-15. OKX API: venue-адаптер + event-реакции (WS-first, луковая архитектура)

> **Статус: ⬜ спека готова, реализация не начата.**
> Выполняется после TZ-03 волны 2 (универсальный маппер индикаторов).
> Скоуп v1: SPOT + SWAP (бессрочные), WS-first, нормализация под канон T-Invest.

## 1. Контекст

Система работает с T-Invest (Z-архитектура TZ-00). OKX — второй venue: источник
свечей (md.ohlcv) и исполнитель решений. Требование: **OKX подводится под канон
T-Invest** — адаптер не вводит своих форматов, а нормализует всё в существующие
контракты TZ-02/TZ-09 (`Candle`, `OhlcvBatch`, `SignalEvent`, `Decision`) на границе.
Данные собираются через WebSocket (пуш), REST — только там, где WS не даёт
истории/снапшотов. Архитектура пакета — луковая (зависимости внутрь), сборка — dishka.

Источник: https://www.okx.com/docs-v5/en/ (v5). Единые эндпоинты для всех типов
инструментов; тип задаётся `instType` (SPOT / SWAP / FUTURES) и `tdMode`
(cash / cross / isolated).

## 2. Принцип WS-first, REST — служебный

| Данные | Транспорт | Почему |
|---|---|---|
| Свечи live, tickers, books5 | **WS public** `wss://ws.okx.com:8443/ws/v5/public` | пуш без поллинга |
| История для warmup / `resolve_history` | REST `GET /market/candles`, `GET /market/history-candles` | WS не отдаёт произвольную глубину |
| Ордера place/amend/cancel | REST `POST /trade/order`, `cancel-order`, `amend-order`; `GET /trade/orders-pending`, `GET /trade/fills` | синхронный ответ, идемпотентность `clOrdId` |
| Fills / positions / balance | **WS private** `wss://ws.okx.com:8443/ws/v5/private` (каналы `orders`, `positions`, `account`) | мгновенные реакции |
| Reconcile после разрыва | REST-снапшоты (`positions`, `orders-pending`, `balance`) | источник истины на resume |

REST-поллинг в стационаре отсутствует; появляется только в degraded-режиме
пока WS переподключается.

## 3. Аутентификация и конфигурация

- REST: заголовки `OK-ACCESS-KEY/SIGN/TIMESTAMP/PASSPHRASE`;
  sign = base64(HMAC-SHA256(ts + method + requestPath + body, secret)).
- WS private: op `login` {apiKey, passphrase, timestamp, sign} — та же подпись
  по ts + method(/users/self/verify) + body.
- Стартовая проверка `GET /account/config`: режим аккаунта и `posMode`
  (net / long_short_mode). **Несовпадение с venue-конфигом = отказ старта**,
  а не автопереключение (детерминизм TZ-00).
- `POST /account/set-leverage` — только при старте инструмента, значение из
  конфига; для SPOT не вызывается.
- Rate-limits: собственный токен-бюджет (stdlib), лимиты OKX на
  candles/orders/WS-подписки учитываются в клиенте.

## 4. Нормализация под канон T-Invest (главное)

Канон: `contracts.Candle{inst_id, ts int ms, open, high, low, close, volume}`,
ts = **время открытия бара**; `OhlcvBatch`; направление long/short.

| OKX | → канон | Правило |
|---|---|---|
| `/market/candles` row `[ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]` (все строки) | `Candle` | float/int; volume: SPOT = `vol` (base ccy); SWAP = `vol` × `ctVal` (контракты → базовый объём) — таблица семантики в `mapping.py` |
| `confirm = 0 / 1` | `CandleClosed` | сигнал строится **только по confirm=1**; unconfirmed (confirm=0) — витрина текущего бара, в DSL не попадает (каузальность TZ-04 п.5) |
| `bar`: 1m/3m/5m/15m/30m/1H/4H/1Dutc/1W… | `BAR_MAP` | одна таблица: канонический id (1m/5m/15m/1h/4h/1d/1w) ↔ OKX bar ↔ t-invest interval; тест на полноту в обе стороны |
| `instId` BTC-USDT / BTC-USDT-SWAP ↔ FIGI t-invest | порт `InstrumentMap` | канонический внутренний символ → venue-native; словарь из конфига/PG (TZ-10) |
| `/public/instruments` `lotSz,tickSz,minSz,ctVal` | `InstrumentSpec` | детерминированное округление цены/размера перед place (floor для sz, tick-выравнивание px) |
| `side` + `posSide`, `tdMode` | из `Decision`/конфига | net: side=buy/sell от direction; long_short: side+posSide; tdMode cash/cross/isolated — venue-конфиг |
| ts строкой ms | int ms | dedupe по (inst_id, ts) — идемпотентность стора уже есть (TZ-10) |

## 5. Луковая архитектура (зависимости направлены внутрь)

```
L4 Внешнее      http_impl.py (niquests), ws_impl.py (websockets) — голый транспорт
L3 Адаптеры     okx_client.py (OKX-протокол: sign, login, каналы → домен),
                repository.py (персистенция → TZ-10 PG)
L2 Use-cases    events.py (event-loop), executor.py (Decision→order),
                collector.py (свечи → md.ohlcv)
L1 Домен        domain.py (OkxEvent, OrderRequest, InstrumentSpec, Position),
                ports.py (Protocol-порты — единственный шов DI)
```

Правила: L1 не импортирует ничего наружу; L2 зависит только от L1; L3 реализует
порты L1; L4 — bytes in/out. Risk Engine и DSL входят в L2 **через порты**
(`RiskGate`, `SignalSource`) — `okx/` не импортирует `risk`/`dsl` напрямую;
связывание делает dishka-сборка в main.

## 6. Порты (L1, `ports.py`) — DI-ключи dishka

```python
class VenueTransport(Protocol):    # голый транспорт (L4 за ним)
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def send(self, payload: dict) -> None: ...
    async def recv(self) -> dict | None: ...
    def request(self, method: str, path: str, body: dict | None) -> dict: ...

class MarketDataSource(Protocol):  # candles_history / subscribe / instruments
class OrderGateway(Protocol):      # place (clOrdId) / cancel / amend / pending
class AccountReader(Protocol):     # balance_available / portfolio_state
class InstrumentMap(Protocol):     # канонический символ <-> venue instId + InstrumentSpec
class EventSink(Protocol):         # push(DomainEvent): md.ohlcv / evt.report / лог
class RiskGate(Protocol):          # check(signal) -> Decision (risk.engine за портом)
class SignalSource(Protocol):      # on_candle(batch) -> list[Signal] (dsl за портом)
class StateStore(Protocol):        # save_fill / last_snapshot (reconcile)
```

## 7. Event-loop (самописный мини-API)

Один тип события `OkxEvent` (msgspec), одна очередь на инструмент (без
параллельных реакций на один инструмент), реестр «событие → реакции»
(конфиг-driven как правила TZ-11):

| Событие | Реакция v1 |
|---|---|
| `CandleClosed` | `SignalSource.on_candle` → `RiskGate.check` → `OrderGateway.place` (+`attachAlgoOrds` с SL/TP от ATR) |
| `OrderFilled` (WS private) | обновить позицию/среднюю, `StateStore.save_fill` |
| `AlgoTriggered` (WS `orders`, ordType conditional/oto) | зафиксировать SL/TP-исполнение |
| `PositionUpdated`/`AccountUpdated` | синк `PortfolioState` (peak_capital, day_pnl → drawdown_stop) |
| `WsDisconnected` | `ReconnectPolicy` (переиспользование TZ-09) → REST-reconcile → resume |

Инварианты (код + тесты, не конфиг): приказ только по подтверждённому событию;
любая реакция проходит через Risk Engine; `clOrdId` идемпотентен; обработка
монотонна; reconcile — источник истины после разрыва (сообщения не теряются).
LLM-контур не имеет импорта и транспортного доступа к модулю (TZ-00 п.1).

## 8. dishka-сборка

По паттерну TZ-08 (Protocol-ключи, ленивые коннекты, без сети на старте):
`okx/di.py` — провайдеры по слоям (Transport/Client/Repository → L3,
EventLoop/Executor/Collector → L2), venue-конфиг из env. Встраивается в
существующие контуры main: white-сборка подключает `collector.py`
(md.ohlcv), локальная — `executor.py`. Тесты подменяют порты fake-классами
через `make_container(FakeTransportProvider())` — тестируется сборка.

## 9. Стек

`niquests` (REST, прецедент dte-dsl) · `websockets` (WS; единственная новая
зависимость — aiohttp отвергнут: тянет весь HTTP-стек ради WS-клиента;
официальный okx-SDK отвергнут: свои абстракции ломают принцип «контракты —
наши msgspec-структуры») · `msgspec` (домен/контракты) · `pyyaml` (venue-конфиг)
· stdlib hmac/hashlib (подписи) · pytest/pytest-asyncio (strict). Без torch —
GPU-изоляция TZ-00 п.4.7. Own rate-limit бюджет (токен-бакет на stdlib).

## 10. Критерии приёмки

- Fake-транспорт e2e: `CandleClosed` → сигнал → `RiskGate.check` (reject —
  приказа нет, событие в аудите) → `place` c `clOrdId`; повтор той же свечи —
  дубль не уходит (идемпотентность).
- WS private e2e: `OrderFilled` → позиция обновлена → `PortfolioState` синк.
- Reconcile e2e: разрыв → reconnect с backoff → REST-снапшот → resume без
  потери событий.
- Нормализация: каждый тип OKX-payload имеет тест «payload → канон == эталон»
  (строки→float, ts ms, confirm, volume по instType).
- posMode mismatch → отказ старта; округление lotSz/tickSz детерминировано.
- ruff/mypy чисто; `uv run --package dte-okx pytest okx/tests` зелёный.

## 11. Связи

- **TZ-04**: executor переиспользует семантику execution (п.0 — один движок);
  reject'и RiskGate попадают в аудит как в бэктесте.
- **TZ-09**: `ReconnectPolicy`/`HeartbeatMonitor` reused; OKX-события — вне
  очередей TZ-09, у белого узла свои протоколы.
- **TZ-10**: `InstrumentMap`-словарь и журнал филлов — в PG (Alembic-миграция
  при необходимости).
- **TZ-11**: RiskGate-порт — единственная точка входа решений; лимиты — данные.

