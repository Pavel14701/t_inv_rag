"""Integration tests for the backtest engine (TZ-04 acceptance)."""

from __future__ import annotations

import numpy as np
import pytest

from backtest.contracts import BacktestConfig
from backtest.engine import run_backtest


@pytest.fixture
def ohlc_arrays() -> dict[str, np.ndarray]:
    """Synthetic OHLCV + ATR for 500 bars."""
    rng = np.random.default_rng(42)
    n = 500
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    # Simple ATR approximation: rolling range
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    atr = np.convolve(tr, np.ones(14) / 14, mode="same")
    atr[:14] = np.nan
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "atr": atr,
    }


class TestEngineIntegration:
    def test_trend_following_produces_trades(
        self, ohlc_arrays: dict[str, np.ndarray]
    ) -> None:
        """A simple trend-following signal should produce trades."""
        close = ohlc_arrays["close"]
        atr = ohlc_arrays["atr"]
        entry = np.zeros(len(close), dtype=bool)
        entry[50] = True
        exit_sig = np.zeros(len(close), dtype=bool)
        cfg = BacktestConfig()
        trades, equity, metrics = run_backtest(
            ohlc_arrays["open"],
            ohlc_arrays["high"],
            ohlc_arrays["low"],
            close,
            atr,
            entry,
            exit_sig,
            cfg,
        )
        assert len(trades) >= 1
        assert len(equity) > 0
        assert metrics.n_trades >= 1

    def test_look_ahead_invariant(
        self, ohlc_arrays: dict[str, np.ndarray]
    ) -> None:
        """Replacing bars > 100 with junk does not change the first trade."""
        close = ohlc_arrays["close"].copy()
        atr = ohlc_arrays["atr"].copy()
        entry = np.zeros(len(close), dtype=bool)
        entry[50] = True
        exit_sig = np.zeros(len(close), dtype=bool)
        cfg = BacktestConfig()

        trades_a, _, _ = run_backtest(
            ohlc_arrays["open"],
            ohlc_arrays["high"],
            ohlc_arrays["low"],
            close,
            atr,
            entry,
            exit_sig,
            cfg,
        )
        # Poison bars > 100
        close_p = close.copy()
        close_p[101:] = 9999.0
        atr_p = atr.copy()
        atr_p[101:] = 0.01
        trades_b, _, _ = run_backtest(
            ohlc_arrays["open"],
            ohlc_arrays["high"],
            ohlc_arrays["low"],
            close_p,
            atr_p,
            entry,
            exit_sig,
            cfg,
        )
        if trades_a and trades_b:
            assert trades_a[0].entry_price == pytest.approx(
                trades_b[0].entry_price, rel=1e-10
            )

    def test_reproducibility(self, ohlc_arrays: dict[str, np.ndarray]) -> None:
        """Same inputs -> identical results (deterministic)."""
        close = ohlc_arrays["close"]
        atr = ohlc_arrays["atr"]
        entry = np.zeros(len(close), dtype=bool)
        entry[50] = True
        entry[200] = True
        exit_sig = np.zeros(len(close), dtype=bool)
        cfg = BacktestConfig()
        t1, e1, m1 = run_backtest(
            ohlc_arrays["open"],
            ohlc_arrays["high"],
            ohlc_arrays["low"],
            close,
            atr,
            entry,
            exit_sig,
            cfg,
        )
        t2, e2, m2 = run_backtest(
            ohlc_arrays["open"],
            ohlc_arrays["high"],
            ohlc_arrays["low"],
            close,
            atr,
            entry,
            exit_sig,
            cfg,
        )
        assert [(t.entry_price, t.exit_price) for t in t1] == [
            (t.entry_price, t.exit_price) for t in t2
        ]
        assert np.array_equal(e1, e2)
        assert m1.total_pnl == m2.total_pnl

    def test_no_signals_no_trades(
        self, ohlc_arrays: dict[str, np.ndarray]
    ) -> None:
        """Zero signals -> zero trades, equity stays flat."""
        close = ohlc_arrays["close"]
        atr = ohlc_arrays["atr"]
        cfg = BacktestConfig()
        trades, equity, m = run_backtest(
            ohlc_arrays["open"],
            ohlc_arrays["high"],
            ohlc_arrays["low"],
            close,
            atr,
            np.zeros(len(close), dtype=bool),
            np.zeros(len(close), dtype=bool),
            cfg,
        )
        assert len(trades) == 0
        assert m.n_trades == 0
        assert equity[0] == equity[-1]
