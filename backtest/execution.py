"""Deterministic execution rules (TZ-04 п.4.1, п.0).

Single source of truth shared by ai/ labels, backtest, and live.
Entry fills at ``open[t+1]`` (never at close of the signal bar).
Exit fills from ``t+1`` onward. TP is a limit order (no slippage);
SL and time exits are market orders (slippage against the position).
SL is checked before TP within the same bar (pessimism).
"""

from __future__ import annotations


def effective_entry_price(
    direction: str, raw_price: float, slippage_pct: float
) -> float:
    """Apply slippage to a market order entry price."""
    if direction == "long":
        return raw_price * (1.0 + slippage_pct)
    return raw_price * (1.0 - slippage_pct)


def effective_exit_price(
    direction: str,
    hit_tp: bool,
    raw_exit_price: float,
    slippage_pct: float,
) -> float:
    """Apply slippage to an exit order price.

    TP is a limit order (no slippage); SL/time exits slip against us.
    """
    if hit_tp:
        return raw_exit_price
    if direction == "long":
        return raw_exit_price * (1.0 - slippage_pct)
    return raw_exit_price * (1.0 + slippage_pct)


def compute_tp_sl(
    entry_price: float,
    direction: str,
    atr_value: float,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
) -> tuple[float, float]:
    """Dynamic TP/SL levels from ATR at entry time (causal, data <= t)."""
    sl_dist = atr_value * sl_mult
    tp_dist = atr_value * tp_mult
    if direction == "long":
        return entry_price - sl_dist, entry_price + tp_dist
    return entry_price + sl_dist, entry_price - tp_dist


def check_exit(
    open_p: float,
    high: float,
    low: float,
    sl_price: float,
    tp_price: float,
    direction: str,
) -> tuple[bool, bool]:
    """Check whether SL or TP is hit on this bar (pessimistic: SL first)."""
    if direction == "long":
        hit_sl = low <= sl_price
        hit_tp = high >= tp_price
    else:
        hit_sl = high >= sl_price
        hit_tp = low <= tp_price
    return hit_tp, hit_sl


def realised_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    size: float,
    commission_pct: float,
) -> float:
    """Net PnL after round-trip commission."""
    gross = (exit_price - entry_price) * size
    if direction == "short":
        gross = -gross
    commission = (entry_price + exit_price) * size * commission_pct
    return gross - commission


def realised_r_multiple(
    direction: str,
    entry_price: float,
    exit_price: float,
    sl_price: float,
    tp_price: float,
) -> float:
    """R-multiple: profit/loss normalised by the initial risk (1R)."""
    if direction == "long":
        risk = entry_price - sl_price
        reward = exit_price - entry_price
    else:
        risk = sl_price - entry_price
        reward = entry_price - exit_price
    if risk == 0:
        return 0.0
    return reward / risk
