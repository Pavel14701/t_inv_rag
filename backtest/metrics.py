"""Backtest performance metrics (TZ-04 п.4.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Aggregated performance statistics for a backtest run."""

    profit_factor: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    n_trades: int
    avg_hold_bars: float
    total_pnl: float


def compute_metrics(
    trades: list,
    equity_curve: np.ndarray,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Compute performance metrics from trades and equity curve.

    Args:
        trades: list of Trade objects (backtest.contracts.Trade).
        equity_curve: array of portfolio equity at each bar.
        periods_per_year: for annualising Sharpe (252 for daily bars).

    Returns:
        PerformanceMetrics with PF, Sharpe, MaxDD, win_rate, etc.

    """
    if not trades:
        return PerformanceMetrics(
            profit_factor=0.0,
            sharpe=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            n_trades=0,
            avg_hold_bars=0.0,
            total_pnl=0.0,
        )

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )
    win_rate = len(wins) / len(trades) if trades else 0.0

    returns = np.diff(equity_curve) / equity_curve[:-1]
    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std()) * np.sqrt(
            periods_per_year
        )
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / peak
    max_dd = float(dd.max()) if len(dd) else 0.0

    avg_hold = float(np.mean([t.bars_held for t in trades]))
    total_pnl = float(pnls.sum())

    return PerformanceMetrics(
        profit_factor=profit_factor,
        sharpe=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        n_trades=len(trades),
        avg_hold_bars=avg_hold,
        total_pnl=total_pnl,
    )
