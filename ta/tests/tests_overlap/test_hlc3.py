# -*- coding: utf-8 -*-
"""Unit tests for HLC3 (High-Low-Close average) module.

Tests cover:
- _hlc3 against reference (simple average)
- offset and fillna
- hlc3_ind with Polars Series
- hlc3_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.overlap.hlc3 import _hlc3, hlc3_ind, hlc3_polars


# -----------------------------------------------------------------------------
# Reference implementation
# -----------------------------------------------------------------------------
def _hlc3_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pure Python reference: (high + low + close) / 3."""
    return (high + low + close) / 3.0


# -----------------------------------------------------------------------------
# Tests for _hlc3
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hlc3_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Compare _hlc3 with pure Python reference."""
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    close = prices_random_walk + np.random.randn(n) * 0.2

    result = _hlc3(high, low, close)
    expected = _hlc3_reference(high, low, close)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hlc3_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using _apply_offset_fillna."""
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    close = prices_random_walk + np.random.randn(n) * 0.2

    offset = 3
    fillna = 0.0

    base = _hlc3(high, low, close, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = _hlc3(high, low, close, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hlc3_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hlc3_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """hlc3_ind should accept Polars Series and return NumPy array."""
    np.random.seed(42)
    n = len(prices_random_walk)
    high_s = pl.Series(prices_random_walk + np.random.randn(n) * 0.5)
    low_s = pl.Series(prices_random_walk - np.random.randn(n) * 0.5)
    close_s = pl.Series(prices_random_walk + np.random.randn(n) * 0.2)

    result = hlc3_ind(high_s, low_s, close_s)
    expected = _hlc3_reference(
        high_s.to_numpy(),
        low_s.to_numpy(),
        close_s.to_numpy(),
    )
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hlc3_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hlc3_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """hlc3_polars should add HLC3 column correctly."""
    np.random.seed(42)
    n = len(df_random_walk)
    close = df_random_walk["close"].to_numpy()
    high_arr = close + np.abs(np.random.randn(n) * 0.5)
    low_arr = close - np.abs(np.random.randn(n) * 0.5)
    df = df_random_walk.with_columns(
        [
            pl.Series("high", high_arr),
            pl.Series("low", low_arr),
        ]
    )
    result_df = hlc3_polars(
        df,
        high_col="high",
        low_col="low",
        close_col="close",
        output_col="HLC3",
    )
    assert "HLC3" in result_df.columns
    assert result_df["HLC3"].dtype == pl.Float64
    assert len(result_df) == len(df)
    expected = _hlc3_reference(high_arr, low_arr, close)
    assert_allclose(
        result_df["HLC3"].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_hlc3_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """hlc3_polars should apply offset and fillna."""
    np.random.seed(42)
    n = len(df_random_walk)
    close = df_random_walk["close"].to_numpy()
    high_arr = close + np.abs(np.random.randn(n) * 0.5)
    low_arr = close - np.abs(np.random.randn(n) * 0.5)
    df = df_random_walk.with_columns(
        [
            pl.Series("high", high_arr),
            pl.Series("low", low_arr),
        ]
    )

    offset = 2
    fillna = 0.0
    result_df = hlc3_polars(
        df,
        high_col="high",
        low_col="low",
        close_col="close",
        offset=offset,
        fillna=fillna,
        output_col="HLC3",
    )

    expected = _hlc3(high_arr, low_arr, close, offset=offset, fillna=fillna)
    assert_allclose(
        result_df["HLC3"].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hlc3_with_nan(prices_with_nan):
    """NaN in input propagates correctly to output."""
    high = prices_with_nan + 1.0
    low = prices_with_nan - 1.0
    close = prices_with_nan
    result = _hlc3(high, low, close)
    # NaN at index 5 appears in all, so result[5] should be NaN
    assert np.isnan(result[5])
    # Other indices should be finite (if no other NaNs)
    assert np.isfinite(result[:5]).all()
    assert np.isfinite(result[6:]).all()


@pytest.mark.overlap
def test_hlc3_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN."""
    high = prices_with_inf + 1.0
    low = prices_with_inf - 1.0
    close = prices_with_inf
    result = _hlc3(high, low, close)
    # Inf at index 5 becomes NaN, so result[5] should be NaN
    assert np.isnan(result[5])
    assert np.isfinite(result[:5]).all()
    assert np.isfinite(result[6:]).all()


@pytest.mark.overlap
def test_hlc3_empty(prices_empty):
    """Empty input returns empty array."""
    result = _hlc3(prices_empty, prices_empty, prices_empty)
    assert result.size == 0


@pytest.mark.overlap
def test_hlc3_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = _hlc3(prices_all_nan, prices_all_nan, prices_all_nan)
    assert np.isnan(result).all()
    result_fill = _hlc3(
        prices_all_nan,
        prices_all_nan,
        prices_all_nan,
        fillna=0.0,
    )
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_hlc3_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    high = prices_extreme + 1.0
    low = prices_extreme - 1.0
    close = prices_extreme
    result = _hlc3(high, low, close)
    assert result is not None


@pytest.mark.overlap
def test_hlc3_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly."""
    np.random.seed(42)
    n = len(df_random_walk)
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    high_arr = close_arr + np.abs(np.random.randn(n) * 0.5)
    low_arr = close_arr - np.abs(np.random.randn(n) * 0.5)
    df = df_random_walk.with_columns(
        [
            pl.Series("close", close_arr),
            pl.Series("high", high_arr),
            pl.Series("low", low_arr),
        ]
    )

    result_df = hlc3_polars(
        df,
        high_col="high",
        low_col="low",
        close_col="close",
        output_col="HLC3",
    )
    hlc3_vals = result_df["HLC3"].to_numpy()
    assert np.isnan(hlc3_vals[5])
    assert np.isfinite(hlc3_vals[:5]).all()
    assert np.isfinite(hlc3_vals[6:]).all()
