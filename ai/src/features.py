"""Feature engineering and label generation for order-block trading.

Provides:
- ATR calculation
- Distance-to-order-block features (supply, demand, strongest)
- A state-machine label generator that marks entry/exit actions
    and outcomes based on a configurable strategy.
"""

from typing import NamedTuple

import numpy as np
import polars as pl

from .datatypes import OrderBlock


def compute_atr(df: pl.DataFrame, period: int = 14) -> np.ndarray:
    """Compute the Average True Range (ATR) for a DataFrame.

    Uses a simple moving average of the true range over ``period`` bars.

    Args:
        df: DataFrame with columns 'high', 'low', 'close'.
        period: Lookback period for the moving average (default 14).

    Returns:
        np.ndarray of shape (len(df),) with ATR values as float32.
        Minimum value is clipped to 1e-6 to avoid division by zero.

    """
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
    """Compute ATR-normalised distances to the nearest supply/demand
    zones and to the strongest order block.

    The function iterates over order blocks and updates distance arrays
    only within each block's active window (from ``start_idx`` to
    ``end_idx``).  This is more efficient than a per-bar loop.

    Args:
        df: DataFrame with at least a close price column.
        order_blocks: List of OrderBlock objects; each must have valid
            ``start_idx``, ``end_idx``, ``zone_low``, ``zone_high``,
            ``strength``, and ``block_type``.
        atr_series: ATR values for each bar (from ``compute_atr``).
        close_col: Name of the close price column (default 'close').

    Returns:
        A tuple of three float32 arrays, each of length ``len(df)``:
        - nearest_supply: distance to closest supply zone (999 if none)
        - nearest_demand: distance to closest demand zone (999 if none)
        - strongest_dist: distance to the strongest block (999 if none)

    """
    n = df.height
    close = df[close_col].to_numpy()
    nearest_supply = np.full(n, 999.0, dtype=np.float32)
    nearest_demand = np.full(n, 999.0, dtype=np.float32)
    strongest_dist = np.full(n, 999.0, dtype=np.float32)
    if not order_blocks:
        return nearest_supply, nearest_demand, strongest_dist
    strongest_block = max(order_blocks, key=lambda ob: ob.strength)
    supply_blocks = [
        ob for ob in order_blocks if ob.block_type.lower() == 'supply'
    ]
    demand_blocks = [
        ob for ob in order_blocks if ob.block_type.lower() == 'demand'
    ]

    def _update_distances(blocks, target_array):
        """Fill ``target_array`` with min distance to any block."""
        for ob in blocks:
            start = max(0, ob.start_idx)
            end = min(n, ob.end_idx + 1)
            if start >= end:
                continue
            zone_mid = (ob.zone_low + ob.zone_high) / 2.0
            dist = (
                np.abs(close[start:end] - zone_mid)
                / atr_series[start:end]
            )
            np.minimum(
                target_array[start:end], dist, out=target_array[start:end]
            )
    _update_distances(supply_blocks, nearest_supply)
    _update_distances(demand_blocks, nearest_demand)
    ob = strongest_block
    start = max(0, ob.start_idx)
    end = min(n, ob.end_idx + 1)
    if start < end:
        zone_mid = (ob.zone_low + ob.zone_high) / 2.0
        dist = (
            np.abs(close[start:end] - zone_mid) / atr_series[start:end]
        )
        np.minimum(
            strongest_dist[start:end], dist, out=strongest_dist[start:end]
        )
    return nearest_supply, nearest_demand, strongest_dist


class PositionState(NamedTuple):
    """Immutable state of a single trading position."""

    active: bool
    direction: str | None
    entry_idx: int
    entry_price: float
    tp_price: float
    sl_price: float


def no_position() -> PositionState:
    """Return a default inactive position state."""
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
    """Determine whether TP or SL was hit during a bar.

    When both are hit in the same bar, the one with the smaller
    distance from the open price is chosen.

    Args:
        bar_open: Opening price of the current bar.
        bar_high: High price of the bar.
        bar_low: Low price of the bar.
        position: The currently active position.

    Returns:
        A tuple (hit_tp, hit_sl) indicating which levels were triggered.

    """
    if position.direction == 'long':
        hit_tp = bar_high >= position.tp_price
        hit_sl = bar_low <= position.sl_price
        if hit_tp and hit_sl:
            dist_to_tp = position.tp_price - bar_open
            dist_to_sl = bar_open - position.sl_price
            return (
                (True, False) if dist_to_tp < dist_to_sl else (False, True)
            )
    else:  # short
        hit_tp = bar_low <= position.tp_price
        hit_sl = bar_high >= position.sl_price
        if hit_tp and hit_sl:
            dist_to_tp = bar_open - position.tp_price
            dist_to_sl = position.sl_price - bar_open
            return (
                (True, False) if dist_to_tp < dist_to_sl else (False, True)
            )
    return hit_tp, hit_sl


def _check_rr(
    direction: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    min_rr: float,
) -> bool:
    """Verify that the risk-reward ratio meets the minimum threshold.

    Args:
        direction: 'long' or 'short'.
        entry_price: Entry price (close of the bar).
        tp_price: Take-profit level (absolute price).
        sl_price: Stop-loss level (absolute price).
        min_rr: Minimum required reward-to-risk ratio.

    Returns:
        True if the trade has a valid risk-reward ratio >= min_rr.

    """
    if (
        direction == 'long'
        and (sl_price >= entry_price or tp_price <= entry_price)
        or direction != 'long'
        and (sl_price <= entry_price or tp_price >= entry_price)
    ):
        return False
    if direction == 'long':
        rr = (tp_price - entry_price) / (entry_price - sl_price)
    else:
        rr = (entry_price - tp_price) / (sl_price - entry_price)
    return rr >= min_rr


def _find_entry_ob(
    bar_idx: int,
    bar_high: float,
    bar_low: float,
    order_blocks: list[OrderBlock],
    use_structure_filter: bool,
    trend_filter: str | None,
) -> OrderBlock | None:
    """Search for the first eligible order block for a potential entry.

    A block is eligible if:
    - Its ``end_idx`` is <= current bar index.
    - The bar's high/low overlaps with the block's zone.
    - Optionally passes structure and trend filters.

    Args:
        bar_idx: Index of the current bar.
        bar_high: High price of the bar.
        bar_low: Low price of the bar.
        order_blocks: List of candidate blocks.
        use_structure_filter: If True, requires ``structure_label`` not None.
        trend_filter: If not None, requires matching ``trend_direction``.

    Returns:
        The first matching OrderBlock, or None.

    """
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
    """Handle the exit of a position when TP or SL is hit.

    Updates the ``action`` array with 2 (exit) at bar ``i`` and
    records the outcome (win/loss or R-multiple) at the entry bar.

    Args:
        i: Current bar index.
        open_p: Open price of the bar.
        high: High price.
        low: Low price.
        position: Current active position.
        use_r_multiple: If True, outcome is stored as R-multiple,
            else as binary (1 = tp hit, 0 = sl hit).
        action: 1D action array (modified in-place).
        outcome: 1D outcome array (modified in-place).

    Returns:
        The new position state (usually no_position() if exit occurred,
        otherwise unchanged).

    """
    hit_tp, hit_sl = _check_exit(open_p, high, low, position)
    if not (hit_tp or hit_sl):
        return position
    action[i] = 2
    if use_r_multiple:
        if position.direction == 'long':
            r_mult = (
                (position.tp_price - position.entry_price)
                / (position.entry_price - position.sl_price)
                if hit_tp
                else (position.sl_price - position.entry_price)
                / (position.entry_price - position.sl_price)
            )
        elif hit_tp:
            r_mult = (position.entry_price - position.tp_price) / (
                position.sl_price - position.entry_price
            )
        else:
            r_mult = (position.entry_price - position.sl_price) / (
                position.sl_price - position.entry_price
            )
        outcome[position.entry_idx] = r_mult
    else:
        outcome[position.entry_idx] = 1 if hit_tp else 0
    return no_position()


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
    """Attempt to open a new position when price touches an order block.

    If a valid order block is found and the RR check passes, marks
    action[i] = 1 and returns the new PositionState.

    Args:
        i: Current bar index.
        high: High price of the bar.
        low: Low price.
        close: Close price (used as entry price).
        tp: Take-profit level for this bar.
        sl: Stop-loss level.
        order_blocks: Available order blocks.
        use_structure_filter: Passed to ``_find_entry_ob``.
        trend_filter: Passed to ``_find_entry_ob``.
        min_rr: Minimum risk-reward ratio.
        action: 1D action array (modified in-place).

    Returns:
        PositionState if entry is triggered, else None.

    """
    ob = _find_entry_ob(
        i, high, low, order_blocks, use_structure_filter, trend_filter
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


def generate_labels_from_strategy(
    df: pl.DataFrame,
    order_blocks: list[OrderBlock],
    min_rr: float = 1 / 3,
    use_r_multiple: bool = False,
    use_structure_filter: bool = False,
    trend_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate action and outcome labels by simulating a strategy.

    The strategy enters when the bar touches a valid order block and the
    risk-reward ratio meets ``min_rr``, using the bar's close as entry
    and the pre-calculated TP/SL levels.  Exit is triggered on the first
    bar where TP or SL is hit.

    Args:
        df: DataFrame with columns 'open','high','low','close','tp','sl'.
        order_blocks: List of OrderBlock objects.
        min_rr: Minimum required reward-to-risk ratio (default 1/3).
        use_r_multiple: If True, outcome stores the realised R-multiple
            (NaN where no trade occurred); otherwise stores binary
            outcome (2 = ignore).
        use_structure_filter: If True, only blocks with a non-None
            structure_label are considered.
        trend_filter: If not None, only blocks with this trend_direction
            are considered.

    Returns:
        A tuple of two 1D numpy arrays:
        - action: integer array with -100 (ignore), 0 (hold), 1 (entry),
            2 (exit).  Hold is never used; bars with no event get -100.
        - outcome: float array.  In binary mode: 1 (win), 0 (loss),
            2 (ignore).  In R-multiple mode: actual R-multiple or NaN.

    """
    n = df.height
    action = np.full(n, -100, dtype=int)
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
            position = _process_exit(
                i,
                open_p[i],
                high[i],
                low[i],
                position,
                use_r_multiple,
                action,
                outcome,
            )
            if not position.active:
                continue
        if not position.active:
            new_pos = _process_entry(
                i,
                high[i],
                low[i],
                close[i],
                tp[i],
                sl[i],
                order_blocks,
                use_structure_filter,
                trend_filter,
                min_rr,
                action,
            )
            if new_pos is not None:
                position = new_pos
    return action, outcome
