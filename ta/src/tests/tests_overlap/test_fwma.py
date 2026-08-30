# -*- coding: utf-8 -*-
"""Unit tests for Fibonacci Weighted Moving Average (FWMA) module.

Tests cover:
- Weight generation (_get_fib_weights) with cache, normalisation, asc/desc
- fwma_numba against reference implementation
- offset and fillna
- fwma_ind with Polars Series
- fwma_polars DataFrame integration
- Short window handling
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose, assert_almost_equal

from ...overlap.fwma import (
    _get_fib_weights,
    fwma_numba,
    fwma_ind,
    fwma_polars,
)
from ..._array_ops import _apply_offset_fillna


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------

def _fwma_reference(
    close: npt.NDArray[np.float64],
    length: int,
    asc: bool = True,
) -> npt.NDArray[np.float64]:
    """Pure Python reference implementation of FWMA."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out

    # Generate weights
    w = np.zeros(length, dtype=np.float64)
    w[0] = 1.0
    if length > 1:
        w[1] = 1.0
        for i in range(2, length):
            w[i] = w[i - 1] + w[i - 2]
    if not asc:
        w = w[::-1]
    w /= w.sum()

    # Convolution
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += close[i - j] * w[length - 1 - j]
        out[i] = acc

    return out


# -----------------------------------------------------------------------------
# Tests for weight generation
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_fwma_weights_sum_to_one() -> None:
    """Test that generated weights sum to 1 for both directions."""
    length = 10
    for asc in (True, False):
        weights = _get_fib_weights(length, asc)
        assert_almost_equal(weights.sum(), 1.0, decimal=6)


@pytest.mark.overlap
def test_fwma_weights_direction() -> None:
    """Test that asc=True gives higher weight to recent values."""
    length = 10
    weights_asc = _get_fib_weights(length, True)
    weights_desc = _get_fib_weights(length, False)
    # For asc, weights should increase towards the end
    assert weights_asc[-1] > weights_asc[0]
    # For desc, weights should decrease towards the end
    assert weights_desc[-1] < weights_desc[0]
    # They should be reversed
    assert np.array_equal(weights_asc, weights_desc[::-1])


@pytest.mark.overlap
def test_fwma_weights_cache() -> None:
    """Test that weights are cached (lru_cache works)."""
    length = 10
    asc = True
    w1 = _get_fib_weights(length, asc)
    w2 = _get_fib_weights(length, asc)
    assert np.array_equal(w1, w2)


# -----------------------------------------------------------------------------
# Tests for fwma_numba
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_fwma_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test fwma_numba against pure Python reference for both directions."""
    close = prices_random_walk
    length = 10

    for asc in (True, False):
        result_numba = fwma_numba(close, length=length, asc=asc)
        expected = _fwma_reference(close, length, asc=asc)
        assert_allclose(result_numba, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_fwma_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test offset and fillna using the real _apply_offset_fillna."""
    close = prices_random_walk
    length = 10
    asc = True
    offset = 3
    fillna = 0.0
    base = fwma_numba(close, length=length, asc=asc, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = fwma_numba(
        close, length=length, asc=asc,
        offset=offset, fillna=fillna
    )
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_fwma_numba_short_window() -> None:
    """Window longer than data returns all NaN (or fillna if provided)."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 5
    result = fwma_numba(close, length=length)
    assert np.isnan(result).all()
    # With fillna
    result_fill = fwma_numba(close, length=length, fillna=0.0)
    # _apply_offset_fillna will replace NaNs with fillna
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_fwma_numba_empty_weights() -> None:
    """Test that length=1 works correctly (should return the series itself)."""
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    length = 1
    result = fwma_numba(close, length=length)
    # For length 1, FWMA is just the close price
    assert_allclose(result, close, rtol=1e-6)


# -----------------------------------------------------------------------------
# Tests for fwma_ind (universal wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_fwma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test fwma_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 10
    asc = True
    result = fwma_ind(s, length=length, asc=asc)
    expected = _fwma_reference(prices_random_walk, length, asc=asc)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for fwma_polars (DataFrame integration)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_fwma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test fwma_polars adds a column correctly."""
    length = 10
    asc = True
    result_df = fwma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        asc=asc,
        output_col='FWMA'
    )
    assert 'FWMA' in result_df.columns
    assert result_df['FWMA'].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk['close'].to_numpy()
    expected = _fwma_reference(close_arr, length, asc=asc)
    assert_allclose(
        result_df['FWMA'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_fwma_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame(
        {'close': [
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0
        ]}
    )
    length = 3
    result_df = fwma_polars(df, close_col='close', length=length)
    expected_col = f'FWMA_{length}'
    assert expected_col in result_df.columns


@pytest.mark.overlap
def test_fwma_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test fwma_polars with offset and fillna."""
    length = 10
    asc = True
    offset = 3
    fillna = 0.0

    result_df = fwma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        asc=asc,
        offset=offset,
        fillna=fillna,
        output_col='FWMA'
    )

    close_arr = df_random_walk['close'].to_numpy()
    expected = fwma_numba(
        close_arr, length=length, asc=asc,
        offset=offset, fillna=fillna
    )
    assert_allclose(
        result_df['FWMA'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_fwma_polars_asc_false(df_random_walk: pl.DataFrame) -> None:
    """Test fwma_polars with asc=False (higher weight to older values)."""
    length = 10
    asc = False
    result_df = fwma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        asc=asc,
        output_col='FWMA_DESC'
    )
    assert 'FWMA_DESC' in result_df.columns
    close_arr = df_random_walk['close'].to_numpy()
    expected = _fwma_reference(close_arr, length, asc=False)
    assert_allclose(
        result_df['FWMA_DESC'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_fwma_numba_with_nan(prices_with_nan):
    """NaN in input propagates through FWMA but
    leaves the window eventually.
    """  # noqa: D403
    length = 5
    result = fwma_numba(prices_with_nan, length=length)
    # NaN at index 5. FWMA with window 5:
    # - indices 0-3: NaN (insufficient data)
    # - index 4: SMA of 0-4 (no NaN) -> finite
    # - indices 5-9: windows that include index 5 -> NaN
    # - index 10 and later: window no longer includes NaN -> finite
    assert np.isnan(result[:4]).all()
    assert np.isfinite(result[4])
    assert np.isnan(result[5:10]).all()  # indices 5-9
    assert np.isfinite(result[10:]).all()


@pytest.mark.overlap
def test_fwma_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    length = 5
    result = fwma_numba(prices_with_inf, length=length)
    # Same as NaN test
    assert np.isnan(result[:4]).all()
    assert np.isfinite(result[4])
    assert np.isnan(result[5:10]).all()
    assert np.isfinite(result[10:]).all()


@pytest.mark.overlap
def test_fwma_numba_empty(prices_empty):
    """Empty input returns empty array."""
    result = fwma_numba(prices_empty, length=5)
    assert result.size == 0


@pytest.mark.overlap
def test_fwma_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = fwma_numba(prices_all_nan, length=5)
    assert np.isnan(result).all()
    result_fill = fwma_numba(prices_all_nan, length=5, fillna=0.0)
    # _apply_offset_fillna replaces all NaNs with fillna
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_fwma_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    length = 5
    result = fwma_numba(prices_extreme, length=length)
    assert result is not None
    # At least some finite values after index length-1
    assert np.isfinite(result[length:]).any()


@pytest.mark.overlap
def test_fwma_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly."""
    # Create a copy of the close column with NaN inserted at index 5
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([pl.Series('close', close_arr)])
    result_df = fwma_polars(
        df_with_nan, close_col='close',
        length=5, output_col='FWMA'
    )
    assert 'FWMA' in result_df.columns
    assert len(result_df) == len(df_random_walk)
    fwma_vals = result_df['FWMA'].to_numpy()
    # Check that indices 5-9 are NaN, indices 10+ are finite
    assert np.isnan(fwma_vals[5:10]).all()
    assert np.isfinite(fwma_vals[10:]).all()
