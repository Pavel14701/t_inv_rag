"""Config-driven risk engine (TZ-11).

``check()`` is a pure function: no IO, no wall-clock inside (date/time are
passed in). It walks the *configured* rule pipeline; none of the rule
logic lives here. Hard invariants (no order without every active rule
passing) are enforced structurally: any ``ok=False`` short-circuits to a
reject decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PortfolioConfig, RiskConfig
from .rules import (
    impl as _impl_import,  # noqa: F401  (registers rules)
)
from .rules.base import RuleResult, RuleState, get


@dataclass(frozen=True, slots=True)
class Signal:
    """A candidate entry signal presented to the risk engine."""

    instrument: str
    entry_price: float
    requested_units: float
    has_sl: bool = False
    sl_price: float | None = None
    has_tp: bool = False
    tp_price: float | None = None


@dataclass(slots=True)
class PortfolioState:
    """Mutable runtime state the engine tracks across calls."""

    capital: float = 0.0
    open_positions: int = 0
    day_start_capital: float = 0.0
    day_pnl: float = 0.0
    peak_capital: float = 0.0


@dataclass(frozen=True, slots=True)
class Decision:
    """Outcome of ``check()``. ``approve=True`` only if every active rule
    passed; ``size`` is the capped order size proposed by the rules.
    """

    approve: bool
    reason: str = ""
    size: float = 0.0


def check(
    signal: Signal,
    state: PortfolioState,
    cfg: RiskConfig,
    portfolio: PortfolioConfig | None = None,
) -> Decision:
    """Evaluate a signal against the configured rule pipeline.

    Args:
        signal: candidate order.
        state: mutable portfolio state (capital, open positions, pnl).
        cfg: validated RiskConfig (rules order + params).
        portfolio: optional override; defaults to ``cfg.portfolio``.

    Returns:
        Decision: reject with reason, or approve with the capped size.

    """
    pcfg = portfolio or cfg.portfolio
    internal = RuleState(
        capital=state.capital,
        open_positions=state.open_positions,
        day_start_capital=state.day_start_capital,
        day_pnl=state.day_pnl,
        peak_capital=state.peak_capital,
    )
    decided_size: float | None = None
    for instance in cfg.engine.rules:
        if not instance.active:
            continue
        fn = get(instance.name)
        result: RuleResult = fn(signal, internal, pcfg, instance.params)
        if not result.ok:
            return Decision(
                approve=False,
                reason=f"{instance.name}: {result.reason}",
                size=0.0,
            )
        if result.size is not None:
            decided_size = result.size
    return Decision(approve=True, size=decided_size or 0.0)
