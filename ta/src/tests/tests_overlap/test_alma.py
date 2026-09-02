# -*- coding: utf-8 -*-
"""Unit tests for Arnaud Legoux Moving Average (ALMA) module.

Tests cover:
- Weight generation (_alma_weights)
- alma_numba_opt against reference implementation
- offset and fillna
- alma_ind with Polars Series
- alma_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme values)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose, assert_almost_equal

from ...overlap.alma import (
    _alma_weights,
    alma_numba_opt,
    alma_ind,
    alma_polars
)
from ..._array_ops import _apply_offset_fillna


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------

def _alma_reference(
    close: npt.NDArray[np.float64],
    length: int,
    sigma: float,
    dist_offset: float,
) -> npt.NDArray[np.float64]:
    """Pure Python reference implementation of ALMA."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out

    # Generate weights
    x = np.arange(length, dtype=np.float64)
    k = dist_offset * (length - 1)
    w = np.exp(-0.5 * ((sigma / length) * (x - k)) ** 2)
    w /= w.sum()

    # Apply convolution (reversed weights for correct alignment)
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
def test_alma_weights_sum_to_one() -> None:
    """Test that generated weights sum to 1."""
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    weights = _alma_weights(length, sigma, dist_offset)
    assert_almost_equal(weights.sum(), 1.0, decimal=6)


@pytest.mark.overlap
def test_alma_weights_cache() -> None:
    """Test that weights are cached (lru_cache works)."""
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    w1 = _alma_weights(length, sigma, dist_offset)
    w2 = _alma_weights(length, sigma, dist_offset)
    # Should be the same object (lru_cache)
    assert w1 is w2


@pytest.mark.overlap
def test_alma_weights_readonly() -> None:
    """Cached weights must be read-only to protect the lru_cache."""
    weights = _alma_weights(9, 6.0, 0.85)
    assert not weights.flags.writeable
    with pytest.raises(ValueError):
        weights[0] = 1.0


@pytest.mark.overlap
def test_alma_numba_opt_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match='must be >= 1'):
        alma_numba_opt(close, length=0)


# -----------------------------------------------------------------------------
# Tests for alma_numba_opt
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_alma_numba_opt_against_reference(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test alma_numba_opt against pure Python reference."""
    close = prices_random_walk
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    result_numba = alma_numba_opt(
        close, length=length, sigma=sigma,
        dist_offset=dist_offset
    )
    expected = _alma_reference(close, length, sigma, dist_offset)
    assert_allclose(result_numba, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_alma_numba_opt_short_window() -> None:
    """Test that window longer than data returns all NaN (or fillna)."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 5
    result = alma_numba_opt(close, length=length)
    assert np.isnan(result).all()
    # With fillna
    fillna = 0.0
    result_fill = alma_numba_opt(close, length=length, fillna=fillna)
    assert (result_fill == fillna).all()


@pytest.mark.overlap
def test_alma_numba_opt_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test offset and fillna using the real _apply_offset_fillna."""
    close = prices_random_walk
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    offset = 2
    fillna = 0.0
    # Base result without offset/fillna
    base = alma_numba_opt(
        close, length=length, sigma=sigma,
        dist_offset=dist_offset, offset=0, fillna=None
    )
    # Apply offset and fillna using the same function as in ALMA
    expected = _apply_offset_fillna(base, offset, fillna)

    result = alma_numba_opt(
        close, length=length, sigma=sigma,
        dist_offset=dist_offset, offset=offset, fillna=fillna
    )
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for alma_ind (universal wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_alma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test alma_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    result = alma_ind(s, length=length, sigma=sigma, dist_offset=dist_offset)
    expected = _alma_reference(prices_random_walk, length, sigma, dist_offset)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for alma_polars (DataFrame integration)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_alma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test alma_polars adds a column correctly."""
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    result_df = alma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        sigma=sigma,
        dist_offset=dist_offset,
        output_col='ALMA'
    )
    assert 'ALMA' in result_df.columns
    assert result_df['ALMA'].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk['close'].to_numpy()
    expected = _alma_reference(close_arr, length, sigma, dist_offset)
    assert_allclose(
        result_df['ALMA'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_alma_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame(
        {'close': [
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0
        ]}
    )
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    result_df = alma_polars(
        df, close_col='close', length=length,
        sigma=sigma, dist_offset=dist_offset
    )
    expected_col = f'ALMA_{length}_{sigma}_{dist_offset}'
    assert expected_col in result_df.columns


@pytest.mark.overlap
def test_alma_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test alma_polars with offset and fillna."""
    length = 9
    sigma = 6.0
    dist_offset = 0.85
    offset = 2
    fillna = 0.0
    result_df = alma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        sigma=sigma,
        dist_offset=dist_offset,
        offset=offset,
        fillna=fillna,
        output_col='ALMA'
    )
    close_arr = df_random_walk['close'].to_numpy()
    expected = alma_numba_opt(
        close_arr,
        length=length,
        sigma=sigma,
        dist_offset=dist_offset,
        offset=offset,
        fillna=fillna
    )
    assert_allclose(
        result_df['ALMA'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_alma_polars_no_output_col(df_random_walk: pl.DataFrame) -> None:
    """Test alma_polars when output_col is not provided."""
    length = 7
    sigma = 5.0
    dist_offset = 0.8
    result_df = alma_polars(
        df_random_walk, close_col='close', length=length,
        sigma=sigma, dist_offset=dist_offset
    )
    expected_col = f'ALMA_{length}_{sigma}_{dist_offset}'
    assert expected_col in result_df.columns
    close_arr = df_random_walk['close'].to_numpy()
    expected = _alma_reference(close_arr, length, sigma, dist_offset)
    assert_allclose(
        result_df[expected_col].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_alma_numba_opt_with_nan(prices_with_nan):
    """NaN in input propagates correctly through ALMA calculation."""  # noqa: D403, E501
    length = 5
    result = alma_numba_opt(
        prices_with_nan, length=length, offset=0, fillna=None,
        nan_policy='ignore',
    )
    # NaN at index 5
    # First valid at index length-1 = 4 (no NaN in window 0-4)
    # Windows 5-9 include index 5 NaN -> NaN
    # Window 10 (6-10) no NaN -> finite
    assert np.isfinite(result[4])          # index 4 is finite
    assert np.isnan(result[5:10]).all()    # indices 5-9 are NaN
    assert np.isfinite(result[10:]).all()  # from 10 onward finite


@pytest.mark.overlap
def test_alma_numba_opt_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    length = 5
    result = alma_numba_opt(
        prices_with_inf, length=length, nan_policy='ignore'
    )
    # Same as with NaN because Inf is replaced with NaN
    assert np.isfinite(result[4])
    assert np.isnan(result[5:10]).all()
    assert np.isfinite(result[10:]).all()


@pytest.mark.overlap
def test_alma_numba_opt_empty(prices_empty):
    """Empty input returns empty array."""
    result = alma_numba_opt(prices_empty, length=5)
    assert result.size == 0


@pytest.mark.overlap
def test_alma_numba_opt_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = alma_numba_opt(prices_all_nan, length=5, nan_policy='ignore')
    assert np.isnan(result).all()
    result_fill = alma_numba_opt(
        prices_all_nan, length=5, fillna=0.0, nan_policy='ignore'
    )
    # _apply_offset_fillna replaces all NaNs with fillna
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_alma_numba_opt_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    length = 5
    result = alma_numba_opt(prices_extreme, length=length, nan_policy='ignore')
    # Should not crash; may contain inf or nan, but at least the function runs.
    assert result is not None


@pytest.mark.overlap
def test_alma_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly."""
    # Create a copy and insert NaN at index 5
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([pl.Series('close', close_arr)])
    result_df = alma_polars(
        df_with_nan, close_col='close',
        length=5, output_col='ALMA', nan_policy='ignore'
    )
    assert 'ALMA' in result_df.columns
    assert len(result_df) == len(df_random_walk)
    alma_vals = result_df['ALMA'].to_numpy()
    # Check that the NaN appears in the output
    # ALMA window length 5,
    # NaN at index 5 will affect windows starting from 5 to 9
    # So indices 5..9 should be NaN
    # But we only check that at least some NaN exists
    assert np.isnan(alma_vals[5:10]).any()


@pytest.mark.overlap
def test_alma_numba_opt_nan_policy_raise() -> None:
    """Input with NaN and default nan_policy='raise' raises ValueError."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match='NaN'):
        alma_numba_opt(data, length=3)


@pytest.mark.overlap
def test_alma_numba_opt_nan_policy_ffill() -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    result = alma_numba_opt(data, length=3, nan_policy='ffill')
    # after warmup all values must be finite
    assert np.isfinite(result[2:]).all()
