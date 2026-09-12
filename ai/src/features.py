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

from .config import RiskConfig
from .datatypes import OrderBlock


def compute_atr(
    df: pl.DataFrame,
    period: int | None = None,
    risk: RiskConfig | None = None,
) -> np.ndarray:
    """Compute the causal Average True Range (ATR) for a DataFrame.

    The ATR at bar ``t`` is the mean of true range over the window
    ``[t - period + 1, t]`` (right-aligned, never looking into the
    future).  For the first ``period - 1`` bars an expanding mean is
    used, so the output has no NaN warm-up region.

    Args:
        df: DataFrame with columns 'high', 'low', 'close'.
        period: Lookback period (default 14; overridden by ``risk``).
        risk: Optional :class:`RiskConfig` supplying ``atr_period`` /
            ``atr_floor`` (configs/ai.yaml).

    Returns:
        np.ndarray of shape (len(df),) with ATR values as float32.
        Minimum value is clipped to 1e-6 to avoid division by zero.

    """
    if period is None:
        period = risk.atr_period if risk is not None else 14
    atr_floor = risk.atr_floor if risk is not None else 1e-6
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Causal (right-aligned) moving average: bar t only sees bars <= t
    csum = np.cumsum(tr)
    atr = np.empty(len(tr), dtype=np.float64)
    for i in range(len(tr)):
        if i < period:
            atr[i] = csum[i] / (i + 1)  # expanding mean warm-up
        else:
            atr[i] = (csum[i] - csum[i - period]) / period
    atr[atr <= 0] = atr_floor
    return atr.astype(np.float32)


def compute_tp_sl(
    df: pl.DataFrame,
    atr: np.ndarray | None = None,
    tp_atr_multiplier: float | None = None,
    sl_atr_multiplier: float | None = None,
    close_col: str = 'close',
    risk: RiskConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute absolute TP/SL levels from the **previous bar's** ATR.

    Using the previous bar's ATR guarantees that the levels for bar
    ``t`` are fully known at the close of bar ``t - 1`` (no look-ahead).
    Levels are computed for a long setup:
    ``tp = close + tp_atr_multiplier * atr_prev`` and
    ``sl = close - sl_atr_multiplier * atr_prev``.  For short setups
    the caller can swap/mirror the two arrays.

    Args:
        df: DataFrame with a close price column.
        atr: Pre-computed causal ATR series (from :func:`compute_atr`).
            If ``None``, it is computed with the configured ATR period.
        tp_atr_multiplier: ATR multiplier for the take-profit distance.
        sl_atr_multiplier: ATR multiplier for the stop-loss distance.
        close_col: Name of the close price column (default 'close').
        risk: Optional :class:`RiskConfig` supplying multipliers and the
            ATR period (configs/ai.yaml). Explicit arguments win.

    Returns:
        A tuple ``(tp, sl)`` of float32 arrays of length ``len(df)``.

    """
    tp_mult = (
        tp_atr_multiplier
        if tp_atr_multiplier is not None
        else (risk.tp_atr_multiplier if risk is not None else 2.0)
    )
    sl_mult = (
        sl_atr_multiplier
        if sl_atr_multiplier is not None
        else (risk.sl_atr_multiplier if risk is not None else 1.5)
    )
    close = df[close_col].to_numpy().astype(np.float64)
    if atr is None:
        atr = compute_atr(df, risk=risk)
    atr_prev = np.roll(atr.astype(np.float64), 1)
    atr_prev[0] = atr[0]  # first bar has no previous bar: use its own ATR
    tp = close + tp_mult * atr_prev
    sl = np.maximum(close - sl_mult * atr_prev, 1e-6)
    return tp.astype(np.float32), sl.astype(np.float32)


def compute_ob_distances(
    df: pl.DataFrame,
    order_blocks: list[OrderBlock],
    atr_series: np.ndarray,
    close_col: str = 'close',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ATR-normalised distances to order blocks per bar.

    For every bar ``t`` only blocks **active at that bar** (``start_idx
    <= t <= end_idx``) are considered, and the "strongest" block is
    selected per bar among the active ones.  This avoids the look-ahead
    of selecting a globally strongest block using future information.

    The function iterates over order blocks and updates distance arrays
    only within each block's active window (from ``start_idx`` to
    ``end_idx``).  This is more efficient than a per-bar loop over all
    blocks.

    Args:
        df: DataFrame with at least a close price column.
        order_blocks: List of OrderBlock objects; each must have valid
            ``start_idx``, ``end_idx``, ``zone_low``, ``zone_high``,
            ``strength``, and ``block_type``.
        atr_series: ATR values for each bar (from ``compute_atr``).
        close_col: Name of the close price column (default 'close').

    Returns:
        A tuple of four float32 arrays, each of length ``len(df)``:
        - nearest_supply: distance to closest active supply zone
            (999 if none)
        - nearest_demand: distance to closest active demand zone
            (999 if none)
        - strongest_dist: distance to the strongest *active* block
            (999 if none)
        - is_in_zone: 1.0 if the close price is inside any active
            block zone, else 0.0

    """
    n = df.height
    close = df[close_col].to_numpy()
    nearest_supply = np.full(n, 999.0, dtype=np.float32)
    nearest_demand = np.full(n, 999.0, dtype=np.float32)
    strongest_dist = np.full(n, 999.0, dtype=np.float32)
    is_in_zone = np.zeros(n, dtype=np.float32)
    strongest_strength = np.full(n, -np.inf, dtype=np.float64)
    if not order_blocks:
        return (
            nearest_supply, nearest_demand, strongest_dist, is_in_zone
        )
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

    def _update_zone_flags(blocks):
        """Set ``is_in_zone`` where close is inside an active zone."""
        for ob in blocks:
            start = max(0, ob.start_idx)
            end = min(n, ob.end_idx + 1)
            if start >= end:
                continue
            inside = (
                (close[start:end] >= ob.zone_low)
                & (close[start:end] <= ob.zone_high)
            )
            is_in_zone[start:end] = np.maximum(
                is_in_zone[start:end], inside.astype(np.float32)
            )

    def _update_strongest(blocks):
        """Track distance to the strongest block active at each bar."""
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
            seg_strength = strongest_strength[start:end]
            take = ob.strength > seg_strength
            seg_strength[take] = ob.strength
            seg_dist = strongest_dist[start:end]
            seg_dist[take] = dist[take]

    _update_distances(supply_blocks, nearest_supply)
    _update_distances(demand_blocks, nearest_demand)
    _update_strongest(list(order_blocks))
    _update_zone_flags(list(order_blocks))
    return (
        nearest_supply, nearest_demand, strongest_dist, is_in_zone
    )


class PositionState(NamedTuple):
    """Immutable state of a single trading position."""

    active: bool
    direction: str | None
    entry_idx: int
    entry_price: float
    tp_price: float
    sl_price: float
    decision_idx: int = -1


def no_position() -> PositionState:
    """Return a default inactive position state."""
    return PositionState(
        active=False,
        direction=None,
        entry_idx=-1,
        entry_price=0.0,
        tp_price=0.0,
        sl_price=0.0,
        decision_idx=-1,
    )


def _check_exit(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    position: PositionState,
) -> tuple[bool, bool]:
    """Determine whether TP or SL was hit during a bar.

    When both are hit in the same bar, the outcome is resolved
    **pessimistically** (SL is assumed to be hit first), so that
    generated labels do not overestimate performance.

    Args:
        bar_open: Opening price of the current bar (kept for API
            compatibility; the tie-break is pessimistic and does not
            depend on distances).
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
            # Pessimistic: assume the stop was hit first
            return (False, True)
    else:  # short
        hit_tp = bar_low <= position.tp_price
        hit_sl = bar_high >= position.sl_price
        if hit_tp and hit_sl:
            # Pessimistic: assume the stop was hit first
            return (False, True)
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


def _effective_entry_price(
    direction: str, raw_price: float, slippage_pct: float
) -> float:
    """Apply slippage to a market order entry price.

    Longs buy at a higher price, shorts sell at a lower price.

    Args:
        direction: 'long' or 'short'.
        raw_price: The raw execution price (e.g. next bar's open).
        slippage_pct: Slippage rate as a fraction (e.g. 0.0005).

    Returns:
        The effective (slipped) entry price.

    """
    if direction == 'long':
        return raw_price * (1.0 + slippage_pct)
    return raw_price * (1.0 - slippage_pct)


def _effective_exit_price(
    direction: str,
    hit_tp: bool,
    raw_exit_price: float,
    slippage_pct: float,
) -> float:
    """Apply slippage to an exit order price.

    Take-profit exits are limit orders and fill without slippage.
    Stop-loss and time-based exits are market/stop orders and slip
    against the position.

    Args:
        direction: 'long' or 'short'.
        hit_tp: Whether the exit is a take-profit fill.
        raw_exit_price: The raw exit price (SL level or bar close).
        slippage_pct: Slippage rate as a fraction.

    Returns:
        The effective (slipped) exit price.

    """
    if hit_tp:
        return raw_exit_price  # limit order: no slippage
    if direction == 'long':
        return raw_exit_price * (1.0 - slippage_pct)
    return raw_exit_price * (1.0 + slippage_pct)


def _realised_r_multiple(
    direction: str | None,
    entry_price: float,
    sl_price: float,
    exit_price: float,
    commission_pct: float,
) -> float:
    """Compute the realised R-multiple of a closed trade net of costs.

    The risk unit is ``|entry_price - sl_price|``.  Trading costs
    (commission on entry and exit notionals) are subtracted from the
    gross profit before dividing by the risk unit.

    Args:
        direction: 'long' or 'short'.
        entry_price: Effective entry price (after slippage).
        sl_price: Stop-loss level (defines the risk unit).
        exit_price: Effective exit price (after slippage if any).
        commission_pct: One-side commission rate as a fraction (e.g.
            0.001 for 0.1%).

    Returns:
        The realised R-multiple (float).

    """
    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return 0.0
    gross = exit_price - entry_price
    if direction == 'short':
        gross = -gross
    costs = commission_pct * (entry_price + exit_price)
    return float((gross - costs) / risk)


def _process_exit(
    i: int,
    open_p: float,
    high: float,
    low: float,
    close_p: float,
    position: PositionState,
    use_r_multiple: bool,
    action: np.ndarray,
    outcome: np.ndarray,
    commission_pct: float,
    slippage_pct: float,
    max_bars_hold: int,
) -> PositionState:
    """Handle the exit of a position on bar ``i``.

    An exit occurs when TP or SL is hit during the bar (pessimistic
    tie-break: SL first) **or** when the position has been held for
    ``max_bars_hold`` bars (time exit at the bar's close).

    Updates the ``action`` array with 2 (exit) at bar ``i`` and records
    the outcome (win/loss or R-multiple) at the position's decision bar.

    Args:
        i: Current bar index.
        open_p: Open price of the bar.
        high: High price of the bar.
        low: Low price of the bar.
        close_p: Close price of the bar (used for time exits).
        position: Current active position.
        use_r_multiple: If True, outcome is stored as the realised
            R-multiple net of costs; else as binary (1 = win, 0 = loss).
        action: 1D action array (modified in-place).
        outcome: 1D outcome array (modified in-place).
        commission_pct: One-side commission rate as a fraction.
        slippage_pct: Slippage rate as a fraction.
        max_bars_hold: Maximum number of bars to hold; 0 disables the
            time-based exit.

    Returns:
        ``no_position()`` if the position was closed, else unchanged.

    """
    hit_tp, hit_sl = _check_exit(open_p, high, low, position)
    bars_held = i - position.entry_idx + 1
    time_exit = max_bars_hold > 0 and bars_held >= max_bars_hold
    if not (hit_tp or hit_sl or time_exit):
        return position

    if hit_tp:
        raw_exit = position.tp_price
    elif hit_sl:
        raw_exit = position.sl_price
    else:
        raw_exit = close_p  # time-based exit at the bar's close
    exit_price = _effective_exit_price(
        position.direction, hit_tp, raw_exit, slippage_pct
    )

    r_mult = _realised_r_multiple(
        position.direction,
        position.entry_price,
        position.sl_price,
        exit_price,
        commission_pct,
    )

    action[i] = 2
    if use_r_multiple:
        outcome[position.decision_idx] = r_mult
    else:
        # Binary outcome: a trade is a win only if it is net profitable
        outcome[position.decision_idx] = 1.0 if r_mult > 0 else 0.0
    return no_position()


def _find_decision(
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
) -> tuple[str | None, OrderBlock | None]:
    """Detect an entry **decision** on bar ``i``.

    A decision is made when the bar touches a valid order block and the
    risk-reward ratio (based on the bar's close, which is known at the
    decision moment) meets ``min_rr``.  The actual fill happens on the
    **next bar's open** (see :func:`generate_labels_from_strategy`).

    Args:
        i: Current bar index (decision bar).
        high: High price of the bar.
        low: Low price of the bar.
        close: Close price of the bar (used for the RR feasibility
            check only).
        tp: Take-profit level attached to this bar.
        sl: Stop-loss level attached to this bar.
        order_blocks: Available order blocks.
        use_structure_filter: Passed to ``_find_entry_ob``.
        trend_filter: Passed to ``_find_entry_ob``.
        min_rr: Minimum risk-reward ratio.

    Returns:
        ``(direction, block)`` where ``direction`` is 'long'/'short'
        (None if no decision) and ``block`` is the matched order block.

    """
    ob = _find_entry_ob(
        i, high, low, order_blocks, use_structure_filter, trend_filter
    )
    if ob is None:
        return None, None
    direction = 'long' if ob.block_type.lower() == 'demand' else 'short'
    if not _check_rr(direction, close, tp, sl, min_rr):
        return None, None
    return direction, ob


def generate_labels_from_strategy(
    df: pl.DataFrame,
    order_blocks: list[OrderBlock],
    min_rr: float | None = None,
    use_r_multiple: bool | None = None,
    use_structure_filter: bool | None = None,
    trend_filter: str | None = None,
    commission_pct: float | None = None,
    slippage_pct: float | None = None,
    max_bars_hold: int | None = None,
    risk: RiskConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate action and outcome labels by simulating a strategy.

    Risk parameters may come from three sources (priority order):
    explicit keyword argument > ``risk`` (RiskConfig from configs/ai.yaml)
    > legacy defaults (identical to the YAML defaults).

    Execution model (look-ahead free):

    - Bar ``i`` is a **decision bar**: the price touches a valid order
        block and the RR check (based on ``close[i]`` and the bar's
        pre-computed TP/SL levels) passes.  ``action[i] = 1``.
    - The fill happens at ``open[i + 1]`` with slippage against the
        position (longs buy higher, shorts sell lower).
    - Exits are checked starting from the fill bar: TP (limit fill,
        no slippage), SL (stop fill, slippage against the position),
        or a time-based exit at the bar's close after
        ``max_bars_hold`` bars.  When TP and SL are both hit within
        the same bar, the SL is assumed to be hit first (pessimistic).
    - Commission (both sides) is subtracted from the realised result.
    - ``outcome`` is recorded at the decision bar: binary win/loss
        (1/0, net of costs) or the realised R-multiple.

    Args:
        df: DataFrame with columns 'open','high','low','close','tp','sl'.
        order_blocks: List of OrderBlock objects.
        min_rr: Minimum required reward-to-risk ratio (default 1/3).
        use_r_multiple: If True, outcome stores the realised R-multiple
            net of costs (NaN where no trade occurred); otherwise
            stores binary outcome (2 = ignore).
        use_structure_filter: If True, only blocks with a non-None
            structure_label are considered.
        trend_filter: If not None, only blocks with this trend_direction
            are considered.
        commission_pct: One-side commission rate as a fraction
            (default 0.001 = 0.1% per side).
        slippage_pct: Slippage rate as a fraction for market/stop
            orders (default 0.0005 = 0.05%).
        max_bars_hold: Maximum bars to hold before a time-based exit
            (default 20; 0 disables).
        risk: Optional :class:`RiskConfig` from configs/ai.yaml supplying
            all of the above; explicit keyword arguments win.

    Returns:
        A tuple of two 1D numpy arrays:
        - action: integer array with -100 (ignore), 0 (hold), 1 (entry
            decision), 2 (exit).  Bars with no event get -100.
        - outcome: float array.  In binary mode: 1 (win), 0 (loss),
            2 (ignore).  In R-multiple mode: realised R-multiple or NaN.

    """
    # ---------- Risk parameter resolution (TZ-06 п.10) ----------
    # explicit kwarg > risk config > legacy default
    _r = risk
    min_rr = (
        min_rr if min_rr is not None
        else (_r.min_rr if _r is not None else 1 / 3)
    )
    use_r_multiple = (
        use_r_multiple if use_r_multiple is not None
        else (_r.use_r_multiple if _r is not None else False)
    )
    use_structure_filter = (
        use_structure_filter if use_structure_filter is not None
        else (_r.use_structure_filter if _r is not None else False)
    )
    if trend_filter is None and _r is not None:
        trend_filter = _r.trend_filter
    commission_pct = (
        commission_pct if commission_pct is not None
        else (_r.commission_pct if _r is not None else 0.001)
    )
    slippage_pct = (
        slippage_pct if slippage_pct is not None
        else (_r.slippage_pct if _r is not None else 0.0005)
    )
    max_bars_hold = (
        max_bars_hold if max_bars_hold is not None
        else (_r.max_bars_hold if _r is not None else 20)
    )

    n = df.height
    action = np.full(n, -100, dtype=int)
    outcome = np.full(n, np.nan if use_r_multiple else 2, dtype=float)
    open_p = df['open'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    tp = df['tp'].to_numpy()
    sl = df['sl'].to_numpy()

    i = 0
    while i < n - 1:  # a decision on the last bar can never be filled
        direction, ob = _find_decision(
            i, high[i], low[i], close[i], tp[i], sl[i],
            order_blocks, use_structure_filter, trend_filter, min_rr,
        )
        if direction is None or ob is None:
            i += 1
            continue

        # ---- Decision made at bar i; fill at open[i + 1] ----
        action[i] = 1
        fill_bar = i + 1
        position = PositionState(
            active=True,
            direction=direction,
            entry_idx=fill_bar,
            entry_price=_effective_entry_price(
                direction, open_p[fill_bar], slippage_pct
            ),
            tp_price=float(tp[i]),
            sl_price=float(sl[i]),
            decision_idx=i,
        )
        # Walk forward until the position closes
        j = fill_bar
        while j < n:
            position = _process_exit(
                j,
                open_p[j],
                high[j],
                low[j],
                close[j],
                position,
                use_r_multiple,
                action,
                outcome,
                commission_pct,
                slippage_pct,
                max_bars_hold,
            )
            if not position.active:
                break
            j += 1
        if position.active:
            break  # data ended with an open trade; no exit label
        i = j + 1  # continue scanning after the exit bar
    return action, outcome
