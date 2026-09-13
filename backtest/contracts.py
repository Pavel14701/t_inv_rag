"""Data contracts for the backtest engine (TZ-04 п.3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Deterministic execution and portfolio parameters."""

    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    max_bars_hold: int = 0  # 0 = unlimited
    seed: int = 42
    instrument: str = ""  # risk-config per-instrument overrides key


@dataclass(frozen=True, slots=True)
class RiskReject:
    """A risk-rejected entry, stored in the report (TZ-11 item 7).

    Keeps the rule name, the reason and the params snapshot at the moment
    of the refusal so reports are auditable after config changes.
    """

    bar_idx: int
    rule: str
    reason: str
    params: dict


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round-trip trade (filled entry + exit)."""

    direction: str
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    size: float
    r_multiple: float
    pnl: float
    bars_held: int
    exit_reason: str
