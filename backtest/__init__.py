"""Backtest package (TZ-04): deterministic bar-by-bar execution engine."""

from backtest.contracts import BacktestConfig, Trade
from backtest.execution import (
    check_exit,
    compute_tp_sl,
    effective_entry_price,
    effective_exit_price,
)
from backtest.metrics import compute_metrics


__all__ = [
    "BacktestConfig",
    "Trade",
    "check_exit",
    "compute_metrics",
    "compute_tp_sl",
    "effective_entry_price",
    "effective_exit_price",
]
