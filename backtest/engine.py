"""Bar-by-bar backtest engine (TZ-04 п.4.3).

Takes OHLCV data, ATR values, and entry/exit signal arrays.
Produces trades + equity curve + performance metrics.
Look-ahead safe by construction: entry fill at open[t+1].
"""

from __future__ import annotations

import numpy as np

from backtest.contracts import BacktestConfig
from backtest.metrics import PerformanceMetrics, compute_metrics
from backtest.portfolio import Portfolio


def run_backtest(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    atr_values: np.ndarray,
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    config: BacktestConfig,
    initial_capital: float = 100_000.0,
    position_pct: float = 0.1,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
) -> tuple[list, np.ndarray, PerformanceMetrics]:
    """Run a bar-by-bar backtest.

    Args:
        open_prices: open prices array.
        high_prices: high prices array.
        low_prices: low prices array.
        close_prices: close prices array.
        atr_values: pre-computed ATR array (causal).
        entry_signals: bool array, True = enter on next bar open.
        exit_signals: bool array, True = force exit at next bar.
        config: backtest configuration.
        initial_capital: starting equity.
        position_pct: fraction of capital per trade.
        sl_mult: SL = sl_mult * ATR.
        tp_mult: TP = tp_mult * ATR.

    Returns:
        (trades, equity_curve, metrics)

    """
    n = len(close_prices)
    portfolio = Portfolio(initial_capital, config)

    for t in range(n - 1):  # need t+1 for fill
        portfolio.process_bar(
            bar_idx=t,
            open_p=float(open_prices[t]),
            high=float(high_prices[t]),
            low=float(low_prices[t]),
            close=float(close_prices[t]),
        )

        if portfolio.has_position:
            if exit_signals[t]:
                # Force exit at next open (approximate: use current close)
                pos = portfolio.position
                if pos:
                    from backtest.execution import realised_pnl

                    pnl = realised_pnl(
                        pos.direction,
                        pos.entry_price,
                        float(close_prices[t]),
                        pos.remaining,
                        config.commission_pct,
                    )
                    portfolio.capital += pnl
                    portfolio.position = None
            continue

        # Entry: signal on bar t -> fill at open[t+1]
        if entry_signals[t] and t + 1 < n:
            atr_val = float(atr_values[t])
            if atr_val > 0 and not np.isnan(atr_val):
                next_open = float(open_prices[t + 1])
                size = (portfolio.capital * position_pct) / next_open
                if size > 0:
                    portfolio.open_position(
                        direction="long",
                        signal_bar=t,
                        next_open=next_open,
                        atr_value=atr_val,
                        size=size,
                        sl_mult=sl_mult,
                        tp_mult=tp_mult,
                    )

    # Close remaining position at the last close
    if portfolio.has_position:
        pos = portfolio.position
        if pos:
            from backtest.execution import realised_pnl

            pnl = realised_pnl(
                pos.direction,
                pos.entry_price,
                float(close_prices[-1]),
                pos.remaining,
                config.commission_pct,
            )
            portfolio.capital += pnl
            portfolio.position = None

    equity = np.array(portfolio.equity_curve)
    metrics = compute_metrics(portfolio.trades, equity)
    return portfolio.trades, equity, metrics
