"""Concrete risk rules (v1). Each is parameterized from config only.

Rules never hardcode thresholds - every number comes from ``params``.
The engine (risk/engine.py) stays rule-agnostic.
"""

from __future__ import annotations

from .base import RuleResult, RuleState, register


@register("require_stop_loss")
def require_stop_loss(signal, state: RuleState, portfolio, params):
    """Reject orders without valid SL/TP above the configured floor.

    Params: sl.min_mult / tp.min_mult (fraction of entry price the
    stop/TP must be worth), sl.use_atr (+ atr_lookback) optional.
    """
    if not signal.has_sl or signal.sl_price is None or not signal.has_tp:
        return RuleResult(False, "entry without SL and/or TP")
    sl_min = float(params.get("sl", {}).get("min_mult", 0.0))
    tp_min = float(params.get("tp", {}).get("min_mult", 0.0))
    if signal.entry_price > 0:
        sl_dist = (
            abs(signal.entry_price - signal.sl_price) / signal.entry_price
        )
        tp_dist = (
            abs(signal.entry_price - signal.tp_price) / signal.entry_price
            if signal.tp_price is not None
            else 0.0
        )
        if sl_dist < sl_min:
            return RuleResult(False, "SL closer than min_mult from entry")
        if tp_dist < tp_min:
            return RuleResult(False, "TP closer than min_mult from entry")
    return RuleResult(True, size=None)


@register("position_limit")
def position_limit(signal, state: RuleState, portfolio, params):
    """Cap order size: fraction of capital x price, and max units.

    Params (max_capital_pct / max_units) override the portfolio-level
    defaults; per-instrument overrides shadow both.
    """
    if state.capital <= 0:
        return RuleResult(False, "no capital")
    ovr = portfolio.instruments.get(signal.instrument)
    if ovr and ovr.max_capital_pct is not None:
        pct = ovr.max_capital_pct
    else:
        pct = params.get("max_capital_pct", portfolio.max_capital_pct)
        pct = float(pct) if pct is not None else portfolio.max_capital_pct
    if ovr and ovr.max_units is not None:
        max_units: float | None = ovr.max_units
    else:
        mu = params.get("max_units")
        max_units = float(mu) if mu is not None else portfolio.max_units
    units_by_capital = state.capital * pct / signal.entry_price
    size = min(units_by_capital, max_units or float("inf"))
    size = min(size, float(signal.requested_units))
    if size <= 1e-9:
        return RuleResult(False, "computed size below one unit")
    return RuleResult(True, size=size)


@register("max_positions")
def max_positions(signal, state: RuleState, portfolio, params):
    """Reject when the number of open positions hits the ceiling."""
    limit = int(params.get("max_open", 1))
    if state.open_positions >= limit:
        return RuleResult(False, f"at max open positions ({limit})")
    return RuleResult(True, size=None)


@register("daily_loss_limit")
def daily_loss_limit(signal, state: RuleState, portfolio, params):
    """Reject all signals once the realized daily loss exceeds the cap."""
    cap_pct = float(params.get("max_daily_loss_pct", 0.0))
    limit = state.day_start_capital * cap_pct
    if state.day_pnl <= -limit:
        return RuleResult(
            False,
            f"daily loss limit reached (pnl={state.day_pnl:.2f})",
        )
    return RuleResult(True, size=None)


@register("drawdown_stop")
def drawdown_stop(signal, state: RuleState, portfolio, params):
    """Reject (and force a pause) when drawdown from the peak is too deep.

    While the pause is active (``pause_bars`` checked signals after the
    trigger) every entry is rejected without re-evaluating the drawdown.
    """
    cap_pct = float(params.get("max_drawdown_pct", 0.0))
    pause_bars = int(params.get("pause_bars", 0))
    if state.dd_pause_remaining > 0:
        state.dd_pause_remaining -= 1
        return RuleResult(
            False,
            f"drawdown pause active "
            f"({state.dd_pause_remaining + 1} checks left)",
        )
    if state.peak_capital <= 0:
        return RuleResult(False, "no peak capital tracked")
    dd = (state.peak_capital - state.capital) / state.peak_capital
    if dd >= cap_pct:
        state.dd_pause_remaining = pause_bars
        return RuleResult(False, f"drawdown stop hit (dd={dd:.2%})")
    return RuleResult(True, size=None)
