"""Bar-by-bar backtest engine (TZ-04 п.4.3).

Takes OHLCV data, ATR values, and entry/exit signal arrays.
Produces trades + equity curve + performance metrics.
Look-ahead safe by construction: entry fill at open[t+1].
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from backtest.contracts import BacktestConfig, RiskReject
from backtest.metrics import PerformanceMetrics, compute_metrics
from backtest.portfolio import Portfolio
from risk.config import RiskConfig
from risk.engine import (
    PortfolioState as RiskPortfolioState,
    Signal as RiskSignal,
    check as risk_check,
)


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
    risk_config: RiskConfig | None = None,
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
        risk_config: optional validated RiskConfig; when provided every
            entry passes through ``risk.engine.check`` (TZ-04 п.4.6 /
            TZ-11 п.4.5: one risk gate for backtest and live). Rejected
            entries are recorded in ``metrics.risk_rejects`` with the
            rule name and params snapshot.

    Returns:
        (trades, equity_curve, metrics)

    """
    n = len(close_prices)
    portfolio = Portfolio(initial_capital, config)

    risk_state: RiskPortfolioState | None = None
    if risk_config is not None:
        risk_state = RiskPortfolioState(
            capital=initial_capital,
            open_positions=0,
            day_start_capital=initial_capital,
            day_pnl=0.0,
            peak_capital=initial_capital,
        )
    rejects: list[RiskReject] = []

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
                if risk_state is not None and risk_config is not None:
                    risk_state.capital = portfolio.capital
                    risk_state.open_positions = (
                        1 if portfolio.has_position else 0
                    )
                    risk_state.day_pnl = portfolio.capital - initial_capital
                    risk_state.peak_capital = max(
                        risk_state.peak_capital, portfolio.capital
                    )
                    decision = risk_check(
                        RiskSignal(
                            instrument=config.instrument,
                            entry_price=next_open,
                            requested_units=size,
                            has_sl=True,
                            sl_price=next_open - sl_mult * atr_val,
                            has_tp=True,
                            tp_price=next_open + tp_mult * atr_val,
                        ),
                        risk_state,
                        risk_config,
                    )
                    if not decision.approve:
                        rule_params: dict = {}
                        for instance in risk_config.engine.rules:
                            if instance.name == decision.rule:
                                rule_params = dict(instance.params)
                                break
                        rejects.append(
                            RiskReject(
                                bar_idx=t,
                                rule=decision.rule,
                                reason=decision.reason,
                                params=rule_params,
                            )
                        )
                        continue
                    size = decision.size
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
    if rejects:
        metrics = replace(metrics, risk_rejects=tuple(rejects))
    return portfolio.trades, equity, metrics
