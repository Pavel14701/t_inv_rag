"""Portfolio management: positions, partial takes, equity (TZ-04 п.4.2).

Supports TP1 (partial 50%) + TP2 (remainder), trailing stop by indicator
callback, max_bars_hold, and equity tracking with unrealised PnL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from backtest.contracts import BacktestConfig, Trade
from backtest.execution import (
    check_exit,
    effective_entry_price,
    effective_exit_price,
    realised_pnl,
    realised_r_multiple,
)


@dataclass(slots=True)
class Position:
    """An open trading position."""

    direction: str
    entry_idx: int
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    size: float
    remaining: float
    tp1_hit: bool = False


@dataclass(slots=True)
class PortfolioResult:
    """Result of a backtest run."""

    trades: list[Trade]
    equity_curve: np.ndarray
    final_equity: float


def _compute_tp_sl(
    entry: float,
    direction: str,
    atr: float,
    sl_mult: float,
    tp_mult: float,
) -> tuple[float, float]:
    """Compute SL and TP levels from ATR at entry."""
    if direction == "long":
        return entry - atr * sl_mult, entry + atr * tp_mult
    return entry + atr * sl_mult, entry - atr * tp_mult


class Portfolio:
    """Manages open positions, processes exits, and tracks equity."""

    def __init__(
        self,
        initial_capital: float,
        cfg: BacktestConfig,
        trailing_fn: Callable[[int, float], float] | None = None,
    ) -> None:
        self.capital = initial_capital
        self.cfg = cfg
        self.trailing_fn = trailing_fn
        self.equity_curve: list[float] = [initial_capital]
        self.trades: list[Trade] = []
        self.position: Position | None = None

    @property
    def has_position(self) -> bool:
        """Whether a position is currently open."""
        return self.position is not None

    def open_position(
        self,
        direction: str,
        signal_bar: int,
        next_open: float,
        atr_value: float,
        size: float,
        sl_mult: float = 1.5,
        tp_mult: float = 2.0,
    ) -> float:
        """Open a new position. Returns effective entry price."""
        entry = effective_entry_price(
            direction, next_open, self.cfg.slippage_pct
        )
        sl, tp = _compute_tp_sl(entry, direction, atr_value, sl_mult, tp_mult)
        tp1 = (entry + tp) / 2
        self.position = Position(
            direction=direction,
            entry_idx=signal_bar + 1,
            entry_price=entry,
            sl_price=sl,
            tp1_price=tp1,
            tp2_price=tp,
            size=size,
            remaining=size,
        )
        return entry

    def process_bar(
        self,
        bar_idx: int,
        open_p: float,
        high: float,
        low: float,
        close: float,
    ) -> None:
        """Process exits for the current bar and update equity."""
        pos = self.position
        if pos is None:
            self.equity_curve.append(self.capital)
            return

        hit_tp2, hit_sl = check_exit(
            open_p, high, low, pos.sl_price, pos.tp2_price, pos.direction
        )
        hit_tp1 = not pos.tp1_hit and (
            (pos.direction == "long" and high >= pos.tp1_price)
            or (pos.direction == "short" and low <= pos.tp1_price)
        )
        bars_held = bar_idx - pos.entry_idx
        max_hold = (
            self.cfg.max_bars_hold > 0 and bars_held >= self.cfg.max_bars_hold
        )

        # TP1: partial close (50%)
        if hit_tp1:
            partial = pos.size * 0.5
            pnl1 = realised_pnl(
                pos.direction,
                pos.entry_price,
                pos.tp1_price,
                partial,
                self.cfg.commission_pct,
            )
            self.capital += pnl1
            pos.remaining -= partial
            pos.tp1_hit = True

        # TP2 or SL or time: close remainder
        exit_reason = None
        exit_price = 0.0
        if hit_sl:
            exit_reason = "sl"
            exit_price = effective_exit_price(
                pos.direction, False, pos.sl_price, self.cfg.slippage_pct
            )
        elif hit_tp2:
            exit_reason = "tp2"
            exit_price = pos.tp2_price
        elif max_hold:
            exit_reason = "time"
            exit_price = close

        if exit_reason:
            pnl2 = realised_pnl(
                pos.direction,
                pos.entry_price,
                exit_price,
                pos.remaining,
                self.cfg.commission_pct,
            )
            self.capital += pnl2
            r = realised_r_multiple(
                pos.direction,
                pos.entry_price,
                exit_price,
                pos.sl_price,
                pos.tp2_price,
            )
            tp1_pnl = (
                realised_pnl(
                    pos.direction,
                    pos.entry_price,
                    pos.tp1_price,
                    pos.size * 0.5,
                    self.cfg.commission_pct,
                )
                if pos.tp1_hit
                else 0.0
            )
            self.trades.append(
                Trade(
                    direction=pos.direction,
                    entry_idx=pos.entry_idx,
                    exit_idx=bar_idx,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size=pos.size,
                    r_multiple=r,
                    pnl=pnl2 + tp1_pnl,
                    bars_held=bars_held,
                    exit_reason=exit_reason,
                )
            )
            self.position = None

        # Trailing stop callback
        if self.position is not None and self.trailing_fn is not None:
            new_sl = self.trailing_fn(bar_idx, close)
            if new_sl is not None:
                if pos.direction == "long" and new_sl > pos.sl_price:
                    pos.sl_price = new_sl
                elif pos.direction == "short" and new_sl < pos.sl_price:
                    pos.sl_price = new_sl

        # Equity = capital + unrealised PnL
        if self.position is not None:
            if pos.direction == "long":
                unrealised = (close - pos.entry_price) * pos.remaining
            else:
                unrealised = (pos.entry_price - close) * pos.remaining
            self.equity_curve.append(self.capital + unrealised)
        else:
            self.equity_curve.append(self.capital)
