"""Unit tests for feature engineering functions.

This module tests the ATR calculation and the distance-to-order-block
feature generation functions.
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl

from ..features import compute_atr, compute_ob_distances
from ..datatypes import OrderBlock


@pytest.mark.unit
def test_compute_atr(sample_dataframe: pl.DataFrame) -> None:
    """Test that ATR is computed correctly and returns a positive array.

    The ATR should have the same length as the input DataFrame, dtype float32,
    and all values should be positive.  The first `period` values should be
    filled (not NaN) because the implementation uses forward-fill.

    Args:
        sample_dataframe: Fixture with OHLCV columns.

    Asserts:
        - Return value is a numpy array of float32.
        - Shape matches the number of rows in the DataFrame.
        - All values are strictly positive.
        - No NaN values in the first 14 elements.

    """
    atr: npt.NDArray[np.float32] = compute_atr(sample_dataframe, period=14)
    assert isinstance(atr, np.ndarray)
    assert atr.shape == (len(sample_dataframe),)
    assert atr.dtype == np.float32
    assert np.all(atr > 0)  # all positive
    # Test that first few values are filled (not NaN)
    assert not np.isnan(atr[:14]).any()


@pytest.mark.unit
def test_compute_ob_distances(
    sample_dataframe: pl.DataFrame,
    sample_order_blocks: list[OrderBlock],
    sample_atr: npt.NDArray[np.float32],
) -> None:
    """Test that distance arrays are computed correctly for
    supply/demand zones.

    The function returns three arrays: distance to nearest supply,
    nearest demand, and distance to the strongest order block.
    Default distance is 999.0 where no block is active.
    Since the fixture contains active order blocks, at least
    some finite distances (< 999.0) should exist.

    Args:
        sample_dataframe: Fixture with close price column.
        sample_order_blocks: Fixture with a list of OrderBlock objects.
        sample_atr: Fixture with ATR values.

    Asserts:
        - All returned arrays have shape (n_rows,).
        - Default value 999.0 appears in both supply and demand arrays.
        - At least one value in each array is less than 999.0 (active blocks).

    """
    dist_supply, dist_demand, dist_strongest = compute_ob_distances(
        sample_dataframe, sample_order_blocks, sample_atr
    )
    n = len(sample_dataframe)
    assert dist_supply.shape == (n,)
    assert dist_demand.shape == (n,)
    assert dist_strongest.shape == (n,)
    # Some values should be 999.0 where no block is active
    assert 999.0 in dist_supply
    assert 999.0 in dist_demand
    # Active blocks exist in fixture, so at least some distances are finite
    assert (dist_supply < 999.0).any()
    assert (dist_demand < 999.0).any()
