"""Shared fixtures for the risk engine test suite."""

from __future__ import annotations

import pytest

from risk.engine import PortfolioState, Signal


@pytest.fixture
def portfolio() -> PortfolioState:
    """A well-funded portfolio with a tracked peak."""
    return PortfolioState(
        capital=100_000.0,
        open_positions=0,
        day_start_capital=100_000.0,
        day_pnl=0.0,
        peak_capital=100_000.0,
    )


@pytest.fixture
def entry() -> Signal:
    """A plausible long entry with valid SL/TP."""
    return Signal(
        instrument="TQBR.FIGI",
        entry_price=100.0,
        requested_units=1000.0,
        has_sl=True,
        sl_price=95.0,
        has_tp=True,
        tp_price=120.0,
    )
