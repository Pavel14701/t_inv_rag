from typing import NamedTuple

import numpy as np
import polars as pl

from .datatypes import OrderBlock


def compute_atr(df: pl.DataFrame, period: int = 14) -> np.ndarray:
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = np.convolve(tr, np.ones(period) / period, mode='same')
    atr[:period] = atr[period]
    atr[atr <= 0] = 1e-6
    return atr.astype(np.float32)


def compute_ob_distances(
    df: pl.DataFrame,
    order_blocks: list[OrderBlock],
    atr_series: np.ndarray,
    close_col: str = 'close',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = df.height
    close = df[close_col].to_numpy()

    supply_blocks = [
        ob for ob in order_blocks
        if ob.block_type.lower() == 'supply'
    ]
    demand_blocks = [
        ob for ob in order_blocks
        if ob.block_type.lower() == 'demand'
    ]
    strongest_block = max(
        order_blocks, key=lambda ob: ob.strength
    ) if order_blocks else None

    nearest_supply = np.full(n, 999.0, dtype=np.float32)
    nearest_demand = np.full(n, 999.0, dtype=np.float32)
    strongest_dist = np.full(n, 999.0, dtype=np.float32)

    if not order_blocks:
        return nearest_supply, nearest_demand, strongest_dist

    for t in range(n):
        price = close[t]
        atr = atr_series[t]

        if supply_blocks:
            dists = []
            for ob in supply_blocks:
                zone_mid = (ob.zone_low + ob.zone_high) / 2.0
                dist = abs(price - zone_mid) / atr
                dists.append(dist)
            nearest_supply[t] = min(dists)

        if demand_blocks:
            dists = []
            for ob in demand_blocks:
                zone_mid = (ob.zone_low + ob.zone_high) / 2.0
                dist = abs(price - zone_mid) / atr
                dists.append(dist)
            nearest_demand[t] = min(dists)

        if strongest_block:
            zone_mid = (
                strongest_block.zone_low + strongest_block.zone_high
            ) / 2.0
            strongest_dist[t] = abs(price - zone_mid) / atr

    return nearest_supply, nearest_demand, strongest_dist


class PositionState(NamedTuple):
    active: bool
    direction: str | None
    entry_idx: int
    entry_price: float
    tp_price: float
    sl_price: float


def no_position() -> PositionState:
    return PositionState(
        active=False,
        direction=None,
        entry_idx=-1,
        entry_price=0.0,
        tp_price=0.0,
        sl_price=0.0,
    )


def _check_exit(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    position: PositionState,
) -> tuple[bool, bool]:
    """Возвращает (hit_tp, hit_sl) для текущего бара."""
    if position.direction == 'long':
        hit_tp = bar_high >= position.tp_price
        hit_sl = bar_low <= position.sl_price
        if hit_tp and hit_sl:
            dist_to_tp = position.tp_price - bar_open
            dist_to_sl = bar_open - position.sl_price
            return (True, False) if dist_to_tp < dist_to_sl else (False, True)
    else:  # short
        hit_tp = bar_low <= position.tp_price
        hit_sl = bar_high >= position.sl_price
        if hit_tp and hit_sl:
            dist_to_tp = bar_open - position.tp_price
            dist_to_sl = position.sl_price - bar_open
            return (True, False) if dist_to_tp < dist_to_sl else (False, True)
    return hit_tp, hit_sl


# ----------------------------------------------------------------------
# Расчёт RR и проверка минимального RR
# ----------------------------------------------------------------------
def _check_rr(
    direction: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    min_rr: float,
) -> bool:
    """Проверяет, что RR >= min_rr и уровни корректны."""
    if (
        direction == 'long'
        and (sl_price >= entry_price or tp_price <= entry_price)
        or direction != 'long'
        and (sl_price <= entry_price or tp_price >= entry_price)
    ):
        return False
    elif direction == 'long':
        rr = (tp_price - entry_price) / (entry_price - sl_price)
    else:
        rr = (entry_price - tp_price) / (sl_price - entry_price)
    return rr >= min_rr


# ----------------------------------------------------------------------
# Поиск подходящего ордерблока для входа
# ----------------------------------------------------------------------
def _find_entry_ob(
    bar_idx: int,
    bar_high: float,
    bar_low: float,
    order_blocks: list[OrderBlock],
    use_structure_filter: bool,
    trend_filter: str | None,
) -> OrderBlock | None:
    for ob in order_blocks:
        if ob.end_idx > bar_idx:
            continue
        if not (
            ob.zone_low <= bar_high <= ob.zone_high
            or ob.zone_low <= bar_low <= ob.zone_high
        ):
            continue
        if use_structure_filter and ob.structure_label is None:
            continue
        if trend_filter is not None and ob.trend_direction != trend_filter:
            continue
        return ob
    return None


# ----------------------------------------------------------------------
# Обработка выхода (возвращает новое состояние и заполняет outcome)
# ----------------------------------------------------------------------
def _process_exit(
    i: int,
    open_p: float,
    high: float,
    low: float,
    position: PositionState,
    use_r_multiple: bool,
    action: np.ndarray,
    outcome: np.ndarray,
) -> PositionState:
    hit_tp, hit_sl = _check_exit(open_p, high, low, position)
    if not (hit_tp or hit_sl):
        return position  # выход не произошёл, состояние не меняется

    action[i] = 2
    if use_r_multiple:
        if position.direction == 'long':
            r = (position.tp_price - position.entry_price) / (
                position.entry_price - position.sl_price
            ) if hit_tp else (
                position.sl_price - position.entry_price
            ) / (
                position.entry_price - position.sl_price
            )
        else:
            r = (
                position.entry_price - position.tp_price
            ) / (position.sl_price - position.entry_price) if hit_tp else (
                position.entry_price - position.sl_price) / (
                    position.sl_price - position.entry_price
            )
        outcome[position.entry_idx] = r
    else:
        outcome[position.entry_idx] = 1 if hit_tp else 0
    return no_position()  # позиция закрыта


# ----------------------------------------------------------------------
# Обработка входа (возвращает новое состояние или None)
# ----------------------------------------------------------------------
def _process_entry(
    i: int,
    high: float,
    low: float,
    close: float,
    tp: float,
    sl: float,
    order_blocks: list[OrderBlock],
    use_structure_filter: bool,
    trend_filter: str | None,
    min_rr: float,
    action: np.ndarray,
) -> PositionState | None:
    ob = _find_entry_ob(
        i, high, low,
        order_blocks, use_structure_filter, trend_filter
    )
    if ob is None:
        return None

    direction = 'long' if ob.block_type.lower() == 'demand' else 'short'
    if not _check_rr(direction, close, tp, sl, min_rr):
        return None

    action[i] = 1
    return PositionState(
        active=True,
        direction=direction,
        entry_idx=i,
        entry_price=close,
        tp_price=tp,
        sl_price=sl,
    )


# ----------------------------------------------------------------------
# Основная функция генерации меток (теперь очень короткая)
# ----------------------------------------------------------------------
def generate_labels_from_strategy(
    df: pl.DataFrame,
    order_blocks: list[OrderBlock],
    min_rr: float = 1 / 3,
    use_r_multiple: bool = False,
    use_structure_filter: bool = False,
    trend_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = df.height
    action = np.zeros(n, dtype=int)
    outcome = np.full(n, np.nan if use_r_multiple else 2, dtype=float)

    open_p = df['open'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    tp = df['tp'].to_numpy()
    sl = df['sl'].to_numpy()

    position = no_position()

    for i in range(n):
        if position.active:
            position = _process_exit(i, open_p[i], high[i], low[i], position,
                                     use_r_multiple, action, outcome)
            # если после выхода позиция закрылась, в этом баре вход не ищем
            if not position.active:
                continue
        if not position.active:
            new_pos = _process_entry(
                i, high[i], low[i], close[i], tp[i], sl[i],
                order_blocks, use_structure_filter, trend_filter,
                min_rr, action
            )
            if new_pos is not None:
                position = new_pos
    return action, outcome
