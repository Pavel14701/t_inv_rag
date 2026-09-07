# -*- coding: utf-8 -*-
"""Unit tests for TOS_STDEVALL indicator."""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...statistics.tos_stdevall import (
    tos_stdevall_numpy,
    tos_stdevall_ind,
    tos_stdevall_polars,
)


# -----------------------------------------------------------------------------
# Tests for tos_stdevall_numpy (numpy-based calculation)
# -----------------------------------------------------------------------------

def test_tos_stdevall_numpy_basic() -> None:
    """Test tos_stdevall_numpy with a simple linear price series."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = tos_stdevall_numpy(prices, length=None, stds=[1, 2], ddof=1)

    lr = result['TOS_STDEVALL_LR']
    # Regression line should equal prices (since perfectly linear)
    assert_allclose(lr, prices, rtol=1e-6)

    # Compute expected bands based on actual std
    stdev = np.std(prices, ddof=1)
    expected_l1 = lr - 1 * stdev
    expected_u1 = lr + 1 * stdev
    assert_allclose(result['TOS_STDEVALL_L_1'], expected_l1, rtol=1e-6)
    assert_allclose(result['TOS_STDEVALL_U_1'], expected_u1, rtol=1e-6)

    expected_l2 = lr - 2 * stdev
    expected_u2 = lr + 2 * stdev
    assert_allclose(result['TOS_STDEVALL_L_2'], expected_l2, rtol=1e-6)
    assert_allclose(result['TOS_STDEVALL_U_2'], expected_u2, rtol=1e-6)


@pytest.mark.statistics
def test_tos_stdevall_numpy_with_length(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test tos_stdevall_numpy with a fixed length on random walk."""
    close = prices_random_walk
    length = 50
    result = tos_stdevall_numpy(close, length=length, stds=[1], ddof=1)

    # Check keys include suffix
    assert f'TOS_STDEVALL_{length}_LR' in result
    assert f'TOS_STDEVALL_{length}_L_1' in result
    assert f'TOS_STDEVALL_{length}_U_1' in result

    # Regression line should have same length as the truncated close
    lr = result[f'TOS_STDEVALL_{length}_LR']
    assert len(lr) == length

    # Check that bands are symmetric around LR
    lower = result[f'TOS_STDEVALL_{length}_L_1']
    upper = result[f'TOS_STDEVALL_{length}_U_1']
    assert_allclose(lr - lower, upper - lr, rtol=1e-6)


@pytest.mark.statistics
def test_tos_stdevall_numpy_multipliers(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test that stds multipliers are correctly applied."""
    close = prices_random_walk[-50:]  # use last 50 for deterministic length
    stds = [0.5, 1.0, 2.0]
    result = tos_stdevall_numpy(close, length=None, stds=stds, ddof=1)

    lr = result['TOS_STDEVALL_LR']
    stdev = np.std(close, ddof=1)

    for m in stds:
        lower_key = f'TOS_STDEVALL_L_{m}'
        upper_key = f'TOS_STDEVALL_U_{m}'
        assert lower_key in result
        assert upper_key in result
        assert_allclose(result[lower_key], lr - m * stdev, rtol=1e-6)
        assert_allclose(result[upper_key], lr + m * stdev, rtol=1e-6)


@pytest.mark.statistics
def test_tos_stdevall_numpy_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna parameters."""
    close = prices_random_walk
    length = 30

    result_no_offset = tos_stdevall_numpy(close, length=length, offset=0)
    result_offset = tos_stdevall_numpy(
        close,
        length=length,
        offset=1,
        fillna=0.0,
    )

    # Check first element of each array is 0.0
    for key in result_offset:
        assert result_offset[key][0] == 0.0

    # Check shifted values match no-offset
    for key in result_offset:
        assert_allclose(
            result_offset[key][1:],
            result_no_offset[key][:-1],
            rtol=1e-6,
        )


@pytest.mark.statistics
def test_tos_stdevall_numpy_error_length() -> None:
    """Test that ValueError is raised when length < 2."""
    prices = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match='length must be >= 2'):
        tos_stdevall_numpy(prices, length=1)

    # Also test that empty or single-element series raises
    with pytest.raises(ValueError, match='Need at least 2 data points'):
        tos_stdevall_numpy(np.array([1.0]), length=None)


# -----------------------------------------------------------------------------
# Tests for tos_stdevall_ind (universal wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_tos_stdevall_ind_with_polars_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test tos_stdevall_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 30
    result = tos_stdevall_ind(s, length=length, stds=[1], ddof=1)

    # Check result type
    assert isinstance(result, dict)
    key = f'TOS_STDEVALL_{length}_LR'
    assert key in result
    assert isinstance(result[key], np.ndarray)
    assert len(result[key]) == length


# -----------------------------------------------------------------------------
# Tests for tos_stdevall_polars (Polars DataFrame integration)
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_tos_stdevall_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test tos_stdevall_polars adds columns to DataFrame."""
    result_df = tos_stdevall_polars(
        df_random_walk,
        close_col='close',
        length=30,
        stds=[1, 2],
        ddof=1,
    )

    # Check that columns are added
    expected_cols = [
        'TOS_STDEVALL_30_LR',
        'TOS_STDEVALL_30_L_1',
        'TOS_STDEVALL_30_U_1',
        'TOS_STDEVALL_30_L_2',
        'TOS_STDEVALL_30_U_2',
    ]
    for col in expected_cols:
        assert col in result_df.columns
        assert result_df[col].dtype == pl.Float64

    # Check that length is preserved
    assert len(result_df) == len(df_random_walk)


@pytest.mark.statistics
def test_tos_stdevall_polars_with_suffix(df_random_walk: pl.DataFrame) -> None:
    """Test custom suffix overrides the automatic suffix."""
    result_df = tos_stdevall_polars(
        df_random_walk,
        close_col='close',
        length=50,
        stds=[1],
        suffix='_custom',
    )

    # Columns should use suffix instead of _50
    expected_cols = [
        'TOS_STDEVALL_custom_LR',
        'TOS_STDEVALL_custom_L_1',
        'TOS_STDEVALL_custom_U_1',
    ]
    for col in expected_cols:
        assert col in result_df.columns

    # The automatic suffix should not appear
    assert 'TOS_STDEVALL_50_LR' not in result_df.columns


@pytest.mark.statistics
def test_tos_stdevall_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test offset and fillna in Polars wrapper."""
    result_df = tos_stdevall_polars(
        df_random_walk,
        close_col='close',
        length=30,
        stds=[1],
        offset=1,
        fillna=0.0,
    )

    col = 'TOS_STDEVALL_30_LR'
    assert result_df[col][0] == 0.0

    # Check that other columns also shifted
    for col in ['TOS_STDEVALL_30_L_1', 'TOS_STDEVALL_30_U_1']:
        assert result_df[col][0] == 0.0


@pytest.mark.statistics
def test_tos_stdevall_polars_without_length(df_random_walk: pl.DataFrame) -> None:
    """Test when length=None (uses all data)."""
    result_df = tos_stdevall_polars(
        df_random_walk,
        close_col='close',
        length=None,
        stds=[1],
    )

    # No suffix in column names
    assert 'TOS_STDEVALL_LR' in result_df.columns
    assert 'TOS_STDEVALL_L_1' in result_df.columns
    assert 'TOS_STDEVALL_U_1' in result_df.columns
    assert len(result_df) == len(df_random_walk)


# -----------------------------------------------------------------------------
# IEEE-754 corner-case tests
# -----------------------------------------------------------------------------

def test_tos_stdevall_numpy_nan_propagates() -> None:
    """NaN in the input yields NaN bands (no polyfit warnings or junk)."""
    prices = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    result = tos_stdevall_numpy(prices, stds=[1], ddof=1)
    for arr in result.values():
        assert arr.shape == (5,)
        assert np.isnan(arr).all()


def test_tos_stdevall_numpy_inf_propagates() -> None:
    """An inf in the input yields NaN bands."""
    prices = np.array([1.0, 2.0, np.inf, 4.0, 5.0], dtype=np.float64)
    result = tos_stdevall_numpy(prices, stds=[1], ddof=1)
    for arr in result.values():
        assert np.isnan(arr).all()


def test_tos_stdevall_numpy_invalid_ddof_raises() -> None:
    """Passing ddof >= number of points raises ValueError."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match='ddof must satisfy'):
        tos_stdevall_numpy(prices, ddof=5)
    with pytest.raises(ValueError, match='ddof must satisfy'):
        tos_stdevall_numpy(prices, length=3, ddof=3)


def test_tos_stdevall_numpy_negative_std_raises() -> None:
    """Negative or non-finite band multipliers are rejected."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match='stds multipliers must be'):
        tos_stdevall_numpy(prices, stds=[1.0, -2.0])
    with pytest.raises(ValueError, match='stds multipliers must be'):
        tos_stdevall_numpy(prices, stds=[float('nan')])