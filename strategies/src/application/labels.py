"""Label generator: DSL signals -> action/outcome (TZ-02 п.2.4).

Bridges the strategy layer to the ML training pipeline. Uses the
backtest execution engine (TZ-04) to simulate trades from DSL signals
and produce action/outcome arrays compatible with
``ai.src.dataset.TradingDataset``.

Action labels: 0=hold, 1=entry, 2=exit.
Outcome labels: R-multiple of the completed trade at the exit bar.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from backtest.contracts import BacktestConfig
from backtest.engine import run_backtest
from strategies.src.application.strategy import Strategy


def generate_labels(
    df: pl.DataFrame,
    strategy: Strategy,
    atr_values: np.ndarray,
    config: BacktestConfig | None = None,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
    position_pct: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Generate action/outcome label arrays from a DSL strategy.

    Evaluates the strategy's DSL expressions bar-by-bar via the
    TaProvider, simulates trades through the backtest engine, and
    produces training labels.

    Args:
        df: OHLCV DataFrame with unified schema (open/high/low/close/volume).
        strategy: validated Strategy with dsl_entry/dsl_exit.
        atr_values: pre-computed ATR (causal, len == len(df)).
        config: optional BacktestConfig.
        sl_mult: SL = sl_mult * ATR at entry.
        tp_mult: TP = tp_mult * ATR at entry.
        position_pct: fraction of capital per trade.

    Returns:
        (action_targets, outcome_targets, trades)
        - action_targets: int8 array, 0=hold, 1=entry, 2=exit.
        - outcome_targets: float64 array, R-multiple at exits, NaN elsewhere.
        - trades: list of Trade objects from the backtest engine.

    """
    from dsl.context import Context
    from dsl.interpreter import Interpreter
    from dsl.parser import parse
    from ta.src.provider import TaProvider

    n = len(df)
    cfg = config or BacktestConfig()

    # --- Phase 1: generate entry/exit signal arrays from DSL ---
    provider = TaProvider(df)
    context = Context([provider])
    entry_ast = parse(strategy.dsl_entry)
    exit_ast = parse(strategy.dsl_exit) if strategy.dsl_exit else None
    entry_interp = Interpreter(context)
    exit_interp = Interpreter(context) if exit_ast else None

    entry_signals = np.zeros(n, dtype=bool)
    exit_signals = np.zeros(n, dtype=bool)
    for t in range(n):
        provider.cursor = t
        try:
            if entry_interp.visit(entry_ast):
                entry_signals[t] = True
        except Exception:
            pass
        if exit_ast and exit_interp:
            try:
                if exit_interp.visit(exit_ast):
                    exit_signals[t] = True
            except Exception:
                pass

    # --- Phase 2: simulate trades through the backtest engine ---
    open_prices = df["open"].to_numpy()
    high_prices = df["high"].to_numpy()
    low_prices = df["low"].to_numpy()
    close_prices = df["close"].to_numpy()

    trades, _equity, _metrics = run_backtest(
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices,
        atr_values=atr_values,
        entry_signals=entry_signals,
        exit_signals=exit_signals,
        config=cfg,
        position_pct=position_pct,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
    )

    # --- Phase 3: map trades to action/outcome label arrays ---
    action_targets = np.zeros(n, dtype=np.int8)
    outcome_targets = np.full(n, np.nan, dtype=np.float64)

    for trade in trades:
        # Entry bar: mark as action=1
        entry_bar = trade.entry_idx - 1  # signal bar (before fill)
        if 0 <= entry_bar < n:
            action_targets[entry_bar] = 1
        # Exit bar: mark as action=2, outcome=R-multiple
        exit_bar = trade.exit_idx
        if 0 <= exit_bar < n:
            action_targets[exit_bar] = 2
            outcome_targets[exit_bar] = trade.r_multiple

    return action_targets, outcome_targets, trades
