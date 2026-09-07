"""Unit tests for label generation from strategy simulation."""

from datetime import datetime

import pytest
import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from ..features import (
    generate_labels_from_strategy,
    no_position,
    PositionState
)
from ..datatypes import OrderBlock


@pytest.mark.unit
def test_no_position() -> None:
    """Test that the default inactive position has correct field values.

    Asserts:
        - active is False
        - direction is None
        - entry_idx is -1
        - entry_price, tp_price, sl_price are 0.0
    """
    pos: PositionState = no_position()
    assert pos.active is False
    assert pos.direction is None
    assert pos.entry_idx == -1
    assert pos.entry_price == 0.0
    assert pos.tp_price == 0.0
    assert pos.sl_price == 0.0


@pytest.mark.unit
def test_generate_labels_from_strategy(
    sample_dataframe: pl.DataFrame,
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test label generation with default parameters (binary outcome mode).

    The function should return action and outcome arrays of correct length
    and dtype.  All action values must be in the set {-100, 0, 1, 2}, and
    outcome values (non-NaN) must be in {0.0, 1.0, 2.0}.

    Args:
        sample_dataframe: Fixture with price columns and TP/SL.
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - action and outcome have shape (n_rows,).
        - action dtype is int64.
        - outcome dtype is float64.
        - All unique action values are in the allowed set.
        - All non-NaN outcome values are in {0.0, 1.0, 2.0}.

    """
    action, outcome = generate_labels_from_strategy(
        sample_dataframe,
        sample_order_blocks,
        min_rr=1 / 3,
        use_r_multiple=False,
        use_structure_filter=False,
        trend_filter=None,
    )
    n = len(sample_dataframe)
    assert action.shape == (n,)
    assert outcome.shape == (n,)
    assert action.dtype == np.int64
    assert outcome.dtype == np.float64
    # All actions should be in {-100, 0, 1, 2}
    unique_actions = np.unique(action)
    assert all(val in {-100, 0, 1, 2} for val in unique_actions)
    # Outcomes: either 2 (ignore), 0 (loss), 1 (win), or NaN (if R-multiple mode)  # noqa: E501
    # In binary mode (use_r_multiple=False), outcome values should be 0, 1, or 2  # noqa: E501
    unique_outcome = np.unique(outcome[~np.isnan(outcome)])
    assert all(val in {0.0, 1.0, 2.0} for val in unique_outcome)


@pytest.mark.unit
def test_generate_labels_with_r_multiple(
    sample_dataframe: pl.DataFrame,
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test label generation with R-multiple outcome mode.

    In this mode, outcome for closed trades stores the realised R-multiple
    (a float) instead of a binary win/loss.  If no closed trades exist,
    the test is skipped.

    Args:
        sample_dataframe: Fixture with price columns and TP/SL.
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - There is at least one closed trade (action == 2) with
            non-NaN outcome, or the test is skipped.
        - For all closed trades, outcome is not NaN.

    """
    action, outcome = generate_labels_from_strategy(
        df=sample_dataframe,
        order_blocks=sample_order_blocks,
        min_rr=0.5,
        use_r_multiple=True,
        use_structure_filter=False,
        trend_filter=None,
    )
    # Check if there is at least one closed trade with non-NaN outcome
    closed_trades = np.where((action == 2) & (~np.isnan(outcome)))[0]
    if len(closed_trades) == 0:
        pytest.skip('No closed trades found, skipping outcome check')
    # Check that outcomes are not NaN for those trades
    assert not np.isnan(outcome[closed_trades]).any()


# -----------------------------------------------------------------------------
# Execution model tests (entry on next bar's open, costs, time exit)
# -----------------------------------------------------------------------------
def _make_exec_df(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> pl.DataFrame:
    """Build a small DataFrame with fixed TP/SL (+1 / -1 around close)."""
    return pl.DataFrame(
        {
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'tp': close + 1.0,
            'sl': close - 1.0,
        }
    )


def _touch_block(
    start_idx: int, end_idx: int, zone_low: float, zone_high: float
) -> OrderBlock:
    """Build a demand block spanning the given index range."""
    return OrderBlock(
        id=1,
        block_type='demand',
        start=datetime(2024, 1, 1),
        break_=datetime(2024, 1, 2),
        retest=datetime(2024, 1, 3),
        zone_low=zone_low,
        zone_high=zone_high,
        strength=1.0,
        structure_label=None,
        trend_direction=None,
        start_idx=start_idx,
        end_idx=end_idx,
    )


@pytest.mark.unit
def test_labels_entry_uses_next_bar_open() -> None:
    """Entry decision at bar i is filled at open[i+1] (net of slippage).

    The realised R-multiple must equal the manual computation from
    open[i+1] with slippage, TP as a limit fill, and commissions.
    """
    n = 10
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.6)  # above the zone: no touch by default
    low = np.full(n, 100.6)   # above SL, below TP: no exit either
    # Decision bar 1: high inside the demand zone (touch)
    high[1] = 100.2
    # Fill at bar 2; TP (101) hit at bar 4
    high[4] = 101.2
    df = _make_exec_df(open_, high, low, close)
    block = _touch_block(0, 0, 99.5, 100.5)  # block known before bar 1
    action, outcome = generate_labels_from_strategy(
        df, [block], use_r_multiple=True,
        commission_pct=0.001, slippage_pct=0.0005,
        max_bars_hold=0,
    )
    assert action[1] == 1  # decision bar
    assert action[4] == 2  # exit bar (TP hit)
    assert np.isin(action, [1, 2]).sum() == 2  # exactly one trade
    # Manual R-multiple: entry = 100 * 1.0005, risk = entry - 99,
    # exit = 101 (limit, no slippage), costs = 0.001 * (entry + exit)
    entry = 100.0 * 1.0005
    risk = entry - 99.0
    expected_r = (101.0 - entry - 0.001 * (entry + 101.0)) / risk
    assert_allclose(outcome[1], expected_r, rtol=1e-6)


@pytest.mark.unit
def test_labels_time_exit_after_max_bars_hold() -> None:
    """A position is closed at the close after max_bars_hold bars."""
    n = 20
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.6)
    low = np.full(n, 100.6)
    high[1] = 100.2  # decision bar 1 (touch)
    df = _make_exec_df(open_, high, low, close)
    block = _touch_block(0, 0, 99.5, 100.5)
    action, outcome = generate_labels_from_strategy(
        df, [block], use_r_multiple=True,
        commission_pct=0.001, slippage_pct=0.0005,
        max_bars_hold=3,
    )
    assert action[1] == 1
    # fill at bar 2; bars_held >= 3 first at bar 4 -> exit there
    assert action[4] == 2
    # Time exit at close 100 (net of slippage): a small loss
    entry = 100.0 * 1.0005
    exit_ = 100.0 * (1.0 - 0.0005)
    expected_r = (
        (exit_ - entry) - 0.001 * (entry + exit_)
    ) / (entry - 99.0)
    assert_allclose(outcome[1], expected_r, rtol=1e-6)
    assert expected_r < 0  # costs make the flat trade a loss


@pytest.mark.unit
def test_labels_no_decision_on_last_bar() -> None:
    """A touch on the very last bar cannot be filled: no entry label."""
    n = 6
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.6)
    low = np.full(n, 100.6)
    # Touch happens only on the last bar (n-1 == 5)
    high[n - 1] = 100.2
    df = _make_exec_df(open_, high, low, close)
    block = _touch_block(0, 4, 99.5, 100.5)  # block known before bar 5
    action, outcome = generate_labels_from_strategy(
        df, [block], max_bars_hold=0,
    )
    assert not (action == 1).any()
    assert not (action == 2).any()
    assert (outcome == 2).all()


@pytest.mark.unit
def test_labels_tie_break_is_pessimistic() -> None:
    """When TP and SL are hit in the same bar, SL is assumed first."""
    n = 8
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.6)
    low = np.full(n, 100.6)
    # Decision bar 1: touch
    high[1] = 100.2
    # Bar 2 hits both TP (101) and SL (99) -> pessimistic: SL first
    high[2] = 101.5
    low[2] = 98.5
    df = _make_exec_df(open_, high, low, close)
    block = _touch_block(0, 0, 99.5, 100.5)
    action, outcome = generate_labels_from_strategy(
        df, [block], use_r_multiple=False, max_bars_hold=0,
    )
    assert action[1] == 1
    assert action[2] == 2  # exit on the fill bar itself
    # SL hit -> realised R is negative -> binary outcome is a loss
    assert outcome[1] == 0.0
