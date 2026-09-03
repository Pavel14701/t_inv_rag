# Техническое задание

Все модели описаны как **dataclasses** от msgspec с декоратором `@msgspec.struct`. Это даёт строгую типизацию, валидацию (опционально), и мгновенную сериализацию/десериализацию в JSON.

транспортный уровень - niquest, везде строгая валидация и ретраи с логами в консоль(не глобально а параметром)

---

## 1. Общие типы и перечисления

```python
from msgspec import struct
from enum import Enum
from typing import Optional, List, Union
from datetime import datetime

# Перечисления (упрощают валидацию)
class InstType(str, Enum):
    SPOT = "SPOT"
    FUTURES = "FUTURES"
    SWAP = "SWAP"
    OPTION = "OPTION"

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    FOK = "fok"
    IOC = "ioc"

class TdMode(str, Enum):
    CROSS = "cross"
    ISOLATED = "isolated"

class PosSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NET = "net"   # для режима net (одна позиция)

class OrdState(str, Enum):
    LIVE = "live"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

# Базовый ответ от OKX
@struct
class ApiResponse[T]:
    code: str                # "0" – успех
    msg: str
    data: List[T]
```

---

## 2. Входящие запросы на создание ордера

### 2.1. Базовый ордер (общие поля)

```python
@struct
class OrderRequest:
    inst_id: str                    # "BTC-USDT-SWAP", "BTC-USDT" и т.д.
    side: Side                      # buy / sell
    td_mode: TdMode                 # cross / isolated
    sz: float                       # размер в контрактах (для фьючей) или в валюте (спот)
    pos_side: Optional[PosSide] = None   # только для фьючерсов (long/short) – если не указан, auto
    reduce_only: bool = False       # только закрытие (фьючерсы)
```

### 2.2. Лимитный ордер

```python
@struct
class LimitOrderRequest(OrderRequest):
    ord_type: OrderType = OrderType.LIMIT   # всегда "limit"
    px: float                               # цена лимита
```

### 2.3. Рыночный ордер

```python
@struct
class MarketOrderRequest(OrderRequest):
    ord_type: OrderType = OrderType.MARKET  # всегда "market"
    # px не нужен
```

### 2.4. Стоп-лимитный / Стоп-рыночный (триггерный)

```python
@struct
class StopLimitOrderRequest(LimitOrderRequest):
    # наследует px, sz, inst_id, side, td_mode
    trigger_px: float                       # цена активации
    trigger_type: str = "last"              # "last", "index", "mark" (по умолчанию last)
    trigger_px_type: str = "fill"           # "fill" – при касании, "last" – при проходе
    ord_type: OrderType = OrderType.LIMIT   # или MARKET – сделать через Union
```

Для стоп-маркета:

```python
@struct
class StopMarketOrderRequest(OrderRequest):
    trigger_px: float
    trigger_type: str = "last"
    trigger_px_type: str = "fill"
    ord_type: OrderType = OrderType.MARKET
```

### 2.5. Ордер с тейк-профитом и стоп-лоссом (встроенные)

```python
@struct
class OrderWithTPSLRequest(OrderRequest):
    ord_type: OrderType = OrderType.LIMIT
    px: float
    tp_trigger_px: Optional[float] = None   # цена активации тейк-профита
    tp_order_px: Optional[float] = None     # цена лимита для тейка (если не указан – маркет)
    sl_trigger_px: Optional[float] = None   # цена активации стопа
    sl_order_px: Optional[float] = None     # цена лимита для стопа
    # Можно также добавить отдельные структуры TP/SL, но проще плоскими полями
```

### 2.6. Трейлинг-стоп (фьючерсы)

```python
@struct
class TrailingStopOrderRequest(OrderRequest):
    ord_type: OrderType = OrderType.MARKET   # обычно маркет
    trail_amount: Optional[float] = None    # абсолютное расстояние (в пунктах)
    trail_percent: Optional[float] = None   # или процентное
    trigger_px_type: str = "last"           # "last", "index"
    # активируется автоматически, если цена движется в нужную сторону
```

---

## 3. Исходящие данные (ответы от API)

### 3.1. Информация об ордере (исполненный, активный)

```python
@struct
class OrderInfo:
    inst_id: str
    ord_id: str
    cl_ord_id: Optional[str] = None        # ваш собственный ID, если передавали
    side: Side
    ord_type: OrderType
    px: float
    sz: float
    fill_sz: float                         # исполненный объём
    avg_px: float                          # средняя цена исполнения
    state: OrdState
    td_mode: TdMode
    pos_side: Optional[PosSide] = None
    ccy: Optional[str] = None              # валюта котировки (для спота)
    fill_notional_usd: Optional[float] = None
    # время
    ctime: datetime
    utime: datetime
```

### 3.2. Позиция (фьючерсы)

```python
@struct
class Position:
    inst_id: str
    pos: float                     # положительное – long, отрицательное – short (в режиме net)
    pos_side: PosSide              # "long" или "short" (в режиме long-short)
    avg_px: float                  # средняя цена входа
    upl: float                     # нереализованный PnL
    upl_ratio: float               # в процентах
    lever: float                   # плечо
    mgn_mode: TdMode               # cross / isolated
    liq_px: float                  # цена ликвидации
    ccy: str                       # валюта маржи
    # дополнительные поля для статистики
    notional_usd: float
    adl: int                       # уровень риска
```

### 3.3. Баланс

```python
@struct
class Balance:
    ccy: str                       # валюта, например "USDT"
    cash_bal: float                # доступный баланс
    eq: float                      # эквити (с учётом открытых позиций)
    upl: float                     # нереализованный PnL
    fixed_bal: float               # заблокировано в ордерах
    avail_bal: float               # доступно для торговли
```

### 3.4. Свечи (OHLCV)

```python
@struct
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    # опционально для фьючей:
    volume_ccy: Optional[float] = None
```

### 3.5. Стакан (Order Book)

```python
@struct
class OrderBookLevel:
    px: float
    sz: float
    count: int

@struct
class OrderBook:
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    ts: datetime
```

### 3.6. Сделка (Trade)

```python
@struct
class Trade:
    inst_id: str
    trade_id: int
    side: Side
    px: float
    sz: float
    ts: datetime
```

---

## 4. Ответы от WebSocket (стриминг)

Для WebSocket данные приходят в виде событий. Универсальная обёртка:

```python
@struct
class WSEvent[T]:
    arg: dict           # { "channel": "tickers", "instId": "BTC-USDT" }
    data: List[T]
    action: str         # "snapshot", "update", "remove" (для стакана)
```

**Конкретные типы данных для каналов:**

### Канал `tickers`

```python
@struct
class Ticker:
    inst_id: str
    last: float
    ask_px: float
    ask_sz: float
    bid_px: float
    bid_sz: float
    open24h: float
    high24h: float
    low24h: float
    vol24h: float
    ts: datetime
```

### Канал `positions` (приватный)

```python
@struct
class PositionUpdate(Position):
    # то же, что и Position, но может содержать дополнительные поля
    pass
```

### Канал `orders` (приватный)

```python
@struct
class OrderUpdate(OrderInfo):
    # та же структура, что и OrderInfo
    pass
```

### Канал `account` (баланс)

```python
@struct
class BalanceUpdate:
    ccy: str
    cash_bal: float
    eq: float
    upl: float
    fixed_bal: float
    avail_bal: float
    ts: datetime
```

---

## 5. Специфические для фьючерсов структуры

### 5.1. Инструмент (информация о контракте)

```python
@struct
class Instrument:
    inst_id: str
    inst_type: InstType
    base_ccy: str
    quote_ccy: str
    settle_ccy: str
    contract_val: float           # стоимость одного контракта
    list_time: datetime
    exp_time: Optional[datetime] = None
    lever: float                  # максимальное плечо
    tick_sz: float                # минимальный шаг цены
    lot_sz: float                 # минимальный размер лота
    min_sz: float
    max_sz: float
```

### 5.2. Финансирование (для бессрочных)

```python
@struct
class FundingRate:
    inst_id: str
    funding_rate: float           # текущая ставка
    funding_time: datetime
    next_funding_time: datetime
```

---

## 6. Интерфейсы методов (аргументы и возврат)

Теперь, когда модели данных описаны, можно определить сигнатуры методов. Примеры для REST:

```python
class OKXClient:
    # ---------- Публичные ----------
    async def get_instruments(self, inst_type: InstType) -> List[Instrument]: ...
    async def get_candles(self, inst_id: str, bar: str = "1m", limit: int = 100) -> List[Candle]: ...
    async def get_orderbook(self, inst_id: str, sz: int = 20) -> OrderBook: ...
    async def get_trades(self, inst_id: str, limit: int = 100) -> List[Trade]: ...
    async def get_funding_rate(self, inst_id: str) -> FundingRate: ...

    # ---------- Приватные ----------
    async def get_balance(self, ccy: Optional[str] = None) -> List[Balance]: ...
    async def get_positions(self, inst_id: Optional[str] = None) -> List[Position]: ...

    # Создание ордера – принимает Union типов (самый гибкий способ)
    async def place_order(self, request: Union[LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest, StopMarketOrderRequest, OrderWithTPSLRequest]) -> OrderInfo: ...

    async def cancel_order(self, inst_id: str, ord_id: str) -> bool: ...
    async def amend_order(self, inst_id: str, ord_id: str, new_sz: Optional[float] = None, new_px: Optional[float] = None) -> OrderInfo: ...
    async def set_leverage(self, inst_id: str, lever: float, mgn_mode: TdMode) -> bool: ...
```

---

## 7. Дополнительные удобные структуры

Для упрощения работы можно добавить **фабричные методы** для создания запросов:

```python
# Пример:
def limit_order(inst_id: str, side: Side, sz: float, px: float, td_mode: TdMode = TdMode.CROSS) -> LimitOrderRequest:
    return LimitOrderRequest(inst_id=inst_id, side=side, sz=sz, px=px, td_mode=td_mode)

def stop_market_order(inst_id: str, side: Side, sz: float, trigger_px: float, td_mode: TdMode = TdMode.CROSS) -> StopMarketOrderRequest:
    return StopMarketOrderRequest(inst_id=inst_id, side=side, sz=sz, trigger_px=trigger_px, td_mode=td_mode)
```

---

## 8. Интеграция с системой

Модули `ai` и `dsl` будут использовать эти модели:

- **`dsl`** будет генерировать сигналы (например, `{"action": "entry", "side": "buy", "price": 25000, "sl": 24500, "tp": 26000}`). Этот сигнал преобразуется в `LimitOrderRequest` (с дополнительными TP/SL).
- **`ai`** будет принимать `Candle` и предсказывать действия, затем вызывать `place_order()`.
- **`ta`** (детектор ордер-блоков) будет возвращать `OrderBlock`, которые можно использовать для установки стоп-лоссов.

---

## 9. Преимущества использования msgspec

- **Скорость**: сериализация/десериализация в 5–10 раз быстрее Pydantic.
- **Лёгкость**: нет зависимостей, кроме самого msgspec.
- **Строгость**: типы проверяются во время работы (если включить `strict=True`).
- **Интеграция с asyncio**: отлично работает с `json` (или `msgspec.json`).
