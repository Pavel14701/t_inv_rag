"""Tests for the label generator (TZ-02 п.2.4)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from strategies.src.application.labels import generate_labels
from strategies.src.application.strategy import Strategy


@pytest.fixture
def ohlc(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(100, 10_000, n).astype(np.float64)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def atr(ohlc: pl.DataFrame) -> np.ndarray:
    """Simple rolling ATR proxy (causal)."""
    high = ohlc["high"].to_numpy()
    low = ohlc["low"].to_numpy()
    close = ohlc["close"].to_numpy()
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    atr = np.convolve(tr, np.ones(14) / 14, mode="same")
    atr[:14] = 2.0  # avoid NaN
    return atr


@pytest.fixture
def strategy() -> Strategy:
    return Strategy(
        id="label-test",
        name="RSI entry",
        description="test",
        dsl_entry="close > open",
        dsl_exit="close < open",
    )


class TestGenerateLabels:
    def test_labels_generated(
        self, ohlc: pl.DataFrame, atr: np.ndarray, strategy: Strategy
    ) -> None:
        actions, outcomes, _trades = generate_labels(
            ohlc,
            strategy,
            atr,
        )
        assert len(actions) == len(ohlc)
        assert len(outcomes) == len(ohlc)
        assert actions.dtype == np.int8
        assert outcomes.dtype == np.float64

    def test_entry_labels_exist(
        self, ohlc: pl.DataFrame, atr: np.ndarray, strategy: Strategy
    ) -> None:
        actions, _, _ = generate_labels(ohlc, strategy, atr)
        entries = np.where(actions == 1)[0]
        assert len(entries) > 0, "expected at least one entry signal"

    def test_exit_labels_follow_entries(
        self, ohlc: pl.DataFrame, atr: np.ndarray, strategy: Strategy
    ) -> None:
        actions, _, _ = generate_labels(ohlc, strategy, atr)
        entries = set(np.where(actions == 1)[0])
        exits = set(np.where(actions == 2)[0])
        for exit_bar in exits:
            # every exit must have a matching earlier entry
            assert any(e < exit_bar for e in entries), (
                f"exit at {exit_bar} without prior entry"
            )

    def test_outcome_at_exit_bars(
        self, ohlc: pl.DataFrame, atr: np.ndarray, strategy: Strategy
    ) -> None:
        _actions, outcomes, trades = generate_labels(ohlc, strategy, atr)
        for trade in trades:
            assert not np.isnan(outcomes[trade.exit_idx])

    def test_look_ahead_invariant(
        self, ohlc: pl.DataFrame, atr: np.ndarray, strategy: Strategy
    ) -> None:
        """Labels on bar t must not change when bars > t are replaced."""
        cutoff = 150
        actions_a, _outcomes_a, _ = generate_labels(ohlc, strategy, atr)
        # Poison bars > cutoff
        poisoned = ohlc.clone()
        poisoned = poisoned.with_columns(
            pl.when(pl.int_range(pl.len()) > cutoff)
            .then(9999.0)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        actions_b, _outcomes_b, _ = generate_labels(poisoned, strategy, atr)
        # Labels up to cutoff must be identical
        np.testing.assert_array_equal(
            actions_a[:cutoff],
            actions_b[:cutoff],
            err_msg="look-ahead detected in action labels",
        )

    def test_compatible_with_trading_dataset(
        self, ohlc: pl.DataFrame, atr: np.ndarray, strategy: Strategy
    ) -> None:
        """Output arrays have the right shapes for TradingDataset."""
        actions, outcomes, _ = generate_labels(ohlc, strategy, atr)
        n = len(ohlc)
        assert actions.shape == (n,)
        assert outcomes.shape == (n,)
        assert set(np.unique(actions)).issubset({0, 1, 2})

    def test_hold_dominates(
        self, ohlc: pl.DataFrame, atr: np.ndarray, strategy: Strategy
    ) -> None:
        """Most bars should be hold (0), not entry/exit."""
        actions, _, _ = generate_labels(ohlc, strategy, atr)
        hold_fraction = (actions == 0).sum() / len(actions)
        assert hold_fraction > 0.5  # trading is sparse by design
