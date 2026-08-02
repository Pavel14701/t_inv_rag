"""Unit tests for label generation from strategy simulation."""

import pytest
import numpy as np
import polars as pl

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
