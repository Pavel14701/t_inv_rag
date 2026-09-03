# -*- coding: utf-8 -*-
"""Unit tests for HL2 (High-Low average) module.

Tests cover:
- _hl2 against reference (simple average)
- offset and fillna
- hl2_ind with Polars Series
- hl2_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.hl2 import _hl2, hl2_ind, hl2_polars
from ..._array_ops import _apply_offset_fillna


# -----------------------------------------------------------------------------
# Reference implementation
# -----------------------------------------------------------------------------
def _hl2_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pure Python reference: (high + low) / 2."""
    return (high + low) * 0.5


# -----------------------------------------------------------------------------
# Tests for _hl2
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hl2_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Compare _hl2 with pure Python reference."""
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    result = _hl2(high, low)
    expected = _hl2_reference(high, low)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hl2_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using _apply_offset_fillna."""
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    offset = 3
    fillna = 0.0
    base = _hl2(high, low, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = _hl2(high, low, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hl2_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hl2_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """hl2_ind should accept Polars Series and return NumPy array."""
    np.random.seed(42)
    n = len(prices_random_walk)
    high_s = pl.Series(prices_random_walk + np.random.randn(n) * 0.5)
    low_s = pl.Series(prices_random_walk - np.random.randn(n) * 0.5)
    result = hl2_ind(high_s, low_s)
    expected = _hl2_reference(high_s.to_numpy(), low_s.to_numpy())
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hl2_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hl2_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """hl2_polars should add HL2 column correctly."""
    np.random.seed(42)
    n = len(df_random_walk)
    close = df_random_walk['close'].to_numpy()
    high_arr = close + np.abs(np.random.randn(n) * 0.5)
    low_arr = close - np.abs(np.random.randn(n) * 0.5)
    df = df_random_walk.with_columns([
        pl.Series('high', high_arr),
        pl.Series('low', low_arr),
    ])
    result_df = hl2_polars(
        df, high_col='high', low_col='low', output_col='HL2'
    )
    assert 'HL2' in result_df.columns
    assert result_df['HL2'].dtype == pl.Float64
    assert len(result_df) == len(df)
    expected = _hl2_reference(high_arr, low_arr)
    assert_allclose(
        result_df['HL2'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_hl2_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """hl2_polars should apply offset and fillna."""
    np.random.seed(42)
    n = len(df_random_walk)
    close = df_random_walk['close'].to_numpy()
    high_arr = close + np.abs(np.random.randn(n) * 0.5)
    low_arr = close - np.abs(np.random.randn(n) * 0.5)
    df = df_random_walk.with_columns([
        pl.Series('high', high_arr),
        pl.Series('low', low_arr),
    ])

    offset = 2
    fillna = 0.0
    result_df = hl2_polars(
        df,
        high_col='high',
        low_col='low',
        offset=offset,
        fillna=fillna,
        output_col='HL2',
    )

    expected = _hl2(high_arr, low_arr, offset=offset, fillna=fillna)
    assert_allclose(
        result_df['HL2'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hl2_with_nan(prices_with_nan):
    """NaN in input propagates correctly to output."""  # noqa: D403
    high = prices_with_nan + 1.0
    low = prices_with_nan - 1.0
    result = _hl2(high, low)
    # NaN at index 5 appears in both, so result[5] should be NaN
    assert np.isnan(result[5])
    # Other indices should be finite (if no other NaNs)
    assert np.isfinite(result[:5]).all()
    assert np.isfinite(result[6:]).all()


@pytest.mark.overlap
def test_hl2_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN."""
    high = prices_with_inf + 1.0
    low = prices_with_inf - 1.0
    result = _hl2(high, low)
    # Inf at index 5 becomes NaN, so result[5] should be NaN
    assert np.isnan(result[5])
    assert np.isfinite(result[:5]).all()
    assert np.isfinite(result[6:]).all()


@pytest.mark.overlap
def test_hl2_empty(prices_empty):
    """Empty input returns empty array."""
    result = _hl2(prices_empty, prices_empty)
    assert result.size == 0


@pytest.mark.overlap
def test_hl2_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = _hl2(prices_all_nan, prices_all_nan)
    assert np.isnan(result).all()
    result_fill = _hl2(prices_all_nan, prices_all_nan, fillna=0.0)
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_hl2_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    high = prices_extreme + 1.0
    low = prices_extreme - 1.0
    result = _hl2(high, low)
    assert result is not None


@pytest.mark.overlap
def test_hl2_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly."""
    np.random.seed(42)
    n = len(df_random_walk)
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    high_arr = close_arr + np.abs(np.random.randn(n) * 0.5)
    low_arr = close_arr - np.abs(np.random.randn(n) * 0.5)
    df = df_random_walk.with_columns([
        pl.Series('close', close_arr),
        pl.Series('high', high_arr),
        pl.Series('low', low_arr),
    ])
    result_df = hl2_polars(
        df, high_col='high', low_col='low', output_col='HL2'
    )
    hl2_vals = result_df['HL2'].to_numpy()
    assert np.isnan(hl2_vals[5])
    assert np.isfinite(hl2_vals[:5]).all()
    assert np.isfinite(hl2_vals[6:]).all()
