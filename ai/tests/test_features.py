"""Unit tests for feature engineering functions.

This module tests the ATR calculation and the distance-to-order-block
feature generation functions.
"""

from datetime import datetime

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ..features import (
    compute_atr,
    compute_ob_distances,
    compute_tp_sl,
)
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
    dist_supply, dist_demand, dist_strongest, is_in_zone = (
        compute_ob_distances(sample_dataframe, sample_order_blocks, sample_atr)
    )
    n = len(sample_dataframe)
    assert dist_supply.shape == (n,)
    assert dist_demand.shape == (n,)
    assert dist_strongest.shape == (n,)
    assert is_in_zone.shape == (n,)
    # Some values should be 999.0 where no block is active
    assert 999.0 in dist_supply
    assert 999.0 in dist_demand
    # Active blocks exist in fixture, so at least some distances are finite
    assert (dist_supply < 999.0).any()
    assert (dist_demand < 999.0).any()
    # is_in_zone is binary 0/1
    assert set(np.unique(is_in_zone)).issubset({0.0, 1.0})


# -----------------------------------------------------------------------------
# Anti-look-ahead tests
# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_compute_atr_is_causal(sample_dataframe: pl.DataFrame) -> None:
    """ATR at bar t must not depend on bars after t.

    Compare the full-series ATR against the ATR recomputed on a
    truncated prefix: they must match exactly (up to float32 noise).
    """
    atr_full = compute_atr(sample_dataframe, period=14)
    for cut in (20, 60, 120, 199):
        atr_prefix = compute_atr(sample_dataframe.slice(0, cut), period=14)
        assert_allclose(
            atr_full[:cut], atr_prefix, rtol=1e-5, atol=1e-6
        )


@pytest.mark.unit
def test_compute_tp_sl_uses_previous_bar_atr(
    sample_dataframe: pl.DataFrame,
) -> None:
    """TP/SL at bar t must be built from ATR of bar t-1."""
    atr = compute_atr(sample_dataframe, period=14)
    tp, sl = compute_tp_sl(
        sample_dataframe, atr=atr,
        tp_atr_multiplier=2.0, sl_atr_multiplier=1.5,
    )
    close = sample_dataframe['close'].to_numpy()
    # Bar 0 uses its own ATR (no previous bar exists)
    assert_allclose(tp[0], close[0] + 2.0 * atr[0], rtol=1e-5)
    assert_allclose(sl[0], close[0] - 1.5 * atr[0], rtol=1e-5)
    # Bars t >= 1 use the previous bar's ATR
    for t in (1, 5, 50, 150):
        assert_allclose(tp[t], close[t] + 2.0 * atr[t - 1], rtol=1e-5)
        assert_allclose(sl[t], close[t] - 1.5 * atr[t - 1], rtol=1e-5)


@pytest.mark.unit
def test_compute_ob_distances_is_in_zone_flag(
    sample_dataframe: pl.DataFrame,
) -> None:
    """is_in_zone is 1.0 exactly where close is inside an active zone."""
    atr = compute_atr(sample_dataframe, period=14)
    close = sample_dataframe['close'].to_numpy()
    # A demand block spanning bars 10..30 with a zone around close[15]
    zone_low = float(close[15]) - 0.5
    zone_high = float(close[15]) + 0.5
    ob = OrderBlock(
        id=99, block_type='demand',
        start=datetime.now(), break_=datetime.now(),
        retest=datetime.now(),
        zone_low=zone_low, zone_high=zone_high,
        strength=5.0, structure_label='valid',
        trend_direction='up', start_idx=10, end_idx=30,
    )
    _, _, _, is_in_zone = compute_ob_distances(
        sample_dataframe, [ob], atr
    )
    for t in (12, 20, 30):  # inside the active window
        expected = 1.0 if zone_low <= close[t] <= zone_high else 0.0
        assert is_in_zone[t] == expected
    assert is_in_zone[5] == 0.0  # before the block exists
    assert is_in_zone[40] == 0.0  # after the block ended


@pytest.mark.unit
def test_compute_ob_distances_strongest_is_per_bar(
    sample_dataframe: pl.DataFrame,
) -> None:
    """The strongest block must be selected per bar, not globally.

    A weak block active early and a strong block active later must not
    let the strong block's distance leak into the weak block's period.
    """
    atr = compute_atr(sample_dataframe, period=14)
    close = sample_dataframe['close'].to_numpy()
    mid = float(close[15])
    weak = OrderBlock(
        id=1, block_type='demand',
        start=datetime.now(), break_=datetime.now(),
        retest=datetime.now(),
        zone_low=mid - 0.5, zone_high=mid + 0.5,
        strength=0.1, structure_label=None, trend_direction=None,
        start_idx=10, end_idx=20,
    )
    strong = OrderBlock(
        id=2, block_type='demand',
        start=datetime.now(), break_=datetime.now(),
        retest=datetime.now(),
        zone_low=mid - 2.0, zone_high=mid + 2.0,
        strength=9.0, structure_label=None, trend_direction=None,
        start_idx=21, end_idx=40,
    )
    _, _, strongest, _ = compute_ob_distances(
        sample_dataframe, [weak, strong], atr
    )
    # Inside the weak block's window the distance must be to the weak
    # zone mid (the strong block is not active there yet)
    expected_weak_dist = np.abs(close[15] - mid) / atr[15]
    assert_allclose(strongest[15], expected_weak_dist, rtol=1e-5)
    # Inside the strong block's window the distance is to its zone mid
    expected_strong_dist = np.abs(close[30] - mid) / atr[30]
    assert_allclose(strongest[30], expected_strong_dist, rtol=1e-5)
