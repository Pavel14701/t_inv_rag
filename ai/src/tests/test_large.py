"""Slow integration tests for large datasets.

These tests run on larger synthetic datasets to verify that the feature
engineering and label generation functions scale correctly and produce
valid outputs without memory or performance issues.
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl

from ..features import (
    compute_atr,
    compute_ob_distances,
    generate_labels_from_strategy
)


@pytest.mark.slow
def test_compute_atr_large(large_dataframe: pl.DataFrame) -> None:
    """Test ATR computation on a large dataset.

    The ATR should be computed without errors and produce a float32 array
    of the correct length with no NaN values.

    Args:
        large_dataframe: Fixture with a large synthetic OHLCV dataset.

    Asserts:
        - ATR array length matches the DataFrame length.
        - No NaN values are present.
        - Dtype is float32.

    """
    atr: npt.NDArray[np.float32] = compute_atr(large_dataframe, period=14)
    assert len(atr) == len(large_dataframe)
    assert not np.isnan(atr).any()
    assert atr.dtype == np.float32


@pytest.mark.slow
def test_compute_ob_distances_large(
    large_dataframe: pl.DataFrame,
    large_order_blocks: list,
) -> None:
    """Test order block distance computation on a large dataset.

    The function should return three distance arrays of the correct length
    without any NaN values.

    Args:
        large_dataframe: Fixture with a large synthetic dataset.
        large_order_blocks: Fixture with a large list of OrderBlock objects.

    Asserts:
        - All three distance arrays have length equal to DataFrame rows.
        - No NaN values are present in any array.

    """
    atr: npt.NDArray[np.float32] = compute_atr(large_dataframe, period=14)
    supply, demand, strongest, is_in_zone = compute_ob_distances(
        large_dataframe, large_order_blocks, atr, close_col='close'
    )
    assert len(supply) == len(large_dataframe)
    assert len(demand) == len(large_dataframe)
    assert len(strongest) == len(large_dataframe)
    assert len(is_in_zone) == len(large_dataframe)
    assert not np.isnan(supply).any()
    assert not np.isnan(demand).any()
    assert not np.isnan(strongest).any()


@pytest.mark.slow
def test_label_generation_large(
    large_dataframe: pl.DataFrame,
    large_order_blocks: list,
) -> None:
    """Test label generation on a large dataset.

    The function should produce action and outcome arrays of the correct
    length, with action values restricted to the allowed set {-100, 0, 1, 2}.

    Args:
        large_dataframe: Fixture with a large synthetic dataset.
        large_order_blocks: Fixture with a large list of OrderBlock objects.

    Asserts:
        - Action and outcome arrays have length equal to DataFrame rows.
        - All unique action values are in the allowed set.

    """
    action, outcome = generate_labels_from_strategy(
        large_dataframe, large_order_blocks, min_rr=0.5
    )
    assert len(action) == len(large_dataframe)
    assert len(outcome) == len(large_dataframe)
    # Check that all action values are allowed
    assert np.isin(np.unique(action), [-100, 0, 1, 2]).all()
