# -*- coding: utf-8 -*-
"""Unit tests for rolling median (MEDIAN) module.

Tests cover:
- Numba core function (_median_numba_core)
- median_numba with offset/fillna
- median_ind with Polars Series
- median_polars DataFrame integration
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.statistics.median import (
    _median_numba_core,
    median_ind,
    median_numba,
    median_polars,
)


# -----------------------------------------------------------------------------
# Numba core tests
# -----------------------------------------------------------------------------


def test_median_numba_core_odd() -> None:
    """Test _median_numba_core with odd window size."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    length = 3  # odd
    result = _median_numba_core(prices, length)

    # First length-1 (2) elements are NaN
    expected = np.full_like(prices, np.nan)
    for i in range(length - 1, len(prices)):
        expected[i] = np.median(prices[i - length + 1 : i + 1])

    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


def test_median_numba_core_even() -> None:
    """Test _median_numba_core with even window size."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    length = 4  # even
    result = _median_numba_core(prices, length)

    expected = np.full_like(prices, np.nan)
    for i in range(length - 1, len(prices)):
        expected[i] = np.median(prices[i - length + 1 : i + 1])

    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


def test_median_numba_core_small_window() -> None:
    """Test that window size 1 returns the price itself (no NaN)."""
    prices = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 1
    result = _median_numba_core(prices, length)

    expected = prices.copy()
    assert_allclose(result, expected, rtol=1e-6)


def test_median_numba_core_empty() -> None:
    """Test empty array."""
    prices = np.array([], dtype=np.float64)
    length = 3
    result = _median_numba_core(prices, length)
    assert result.shape == (0,)
    assert result.dtype == np.float64


# -----------------------------------------------------------------------------
# median_numba tests
# -----------------------------------------------------------------------------


@pytest.mark.statistics
def test_median_numba_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test median_numba on a random walk."""
    close = prices_random_walk
    length = 30
    result = median_numba(close, length=length)

    assert result.shape == close.shape
    assert result.dtype == np.float64
    # First length-1 values are NaN
    assert np.isnan(result[: length - 1]).all()
    # After that, values should be finite
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.statistics
def test_median_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 30
    result_no_offset = median_numba(close, length=length, offset=0)
    result_offset = median_numba(close, length=length, offset=1, fillna=0.0)

    assert result_offset[0] == 0.0
    # fillna also replaces the warm-up NaNs of the shifted series
    no_offset_tail = result_no_offset[:-1]
    expected = np.where(np.isnan(no_offset_tail), 0.0, no_offset_tail)
    assert_allclose(result_offset[1:], expected, rtol=1e-6)


@pytest.mark.statistics
def test_median_numba_different_lengths() -> None:
    """Test median_numba with different window lengths."""
    prices = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64
    )

    for length in [2, 3, 4, 5]:
        result = median_numba(prices, length=length)
        expected = np.full_like(prices, np.nan)
        for i in range(length - 1, len(prices)):
            expected[i] = np.median(prices[i - length + 1 : i + 1])
        assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# median_ind tests
# -----------------------------------------------------------------------------


@pytest.mark.statistics
def test_median_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test median_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 30
    result = median_ind(s, length=length)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(prices_random_walk),)
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


# -----------------------------------------------------------------------------
# median_polars tests
# -----------------------------------------------------------------------------


@pytest.mark.statistics
def test_median_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test median_polars adds a column correctly."""
    length = 30
    result_series = median_polars(
        df_random_walk,
        close_col="close",
        length=length,
        output_col="MEDIAN",
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == "MEDIAN"
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk["close"].to_numpy()
    expected = median_numba(close_arr, length=length)
    assert_allclose(
        result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.statistics
def test_median_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    length = 3
    result_series = median_polars(df, close_col="close", length=length)
    assert result_series.name == f"MEDIAN_{length}"


@pytest.mark.statistics
def test_median_polars_with_offset_fillna(
    df_random_walk: pl.DataFrame,
) -> None:
    """Test median_polars with offset and fillna."""
    length = 30
    result_series = median_polars(
        df_random_walk,
        close_col="close",
        length=length,
        offset=1,
        fillna=0.0,
        output_col="MEDIAN",
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk["close"].to_numpy()
    expected = median_numba(close_arr, length=length, offset=1, fillna=0.0)
    assert_allclose(
        result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE-754 corner-case tests
# -----------------------------------------------------------------------------


@pytest.mark.statistics
def test_median_numba_nan_window_is_nan() -> None:
    """A window containing NaN yields NaN (np.median parity).

    Without the finite check, partition() pushes NaN to the end and the
    median of [2, nan, 4] would silently come out as 4.0.
    """
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = median_numba(close, length=3)
    assert np.isnan(result[2:5]).all()
    assert result[5] == pytest.approx(5.0)  # median of [4, 5, 6]


@pytest.mark.statistics
def test_median_numba_inf_window_is_nan() -> None:
    """A window containing +-inf yields NaN, later windows recover."""
    close = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = median_numba(close, length=4)
    # Windows [0..3] and [1..4] contain the inf at index 1.
    assert np.isnan(result[3:5]).all()
    assert result[5] == pytest.approx(4.5)  # median of [3, 4, 5, 6]


@pytest.mark.statistics
def test_median_numba_numpy_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Median values match np.median for every full window."""
    close = prices_random_walk
    length = 21
    result = median_numba(close, length=length)
    for i in range(length - 1, len(close), 13):
        window = close[i - length + 1 : i + 1]
        assert result[i] == pytest.approx(np.median(window), rel=1e-12)


@pytest.mark.statistics
def test_median_numba_invalid_length_raises() -> None:
    """Passing length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="length must be >= 1"):
        median_numba(np.array([1.0, 2.0, 3.0]), length=0)
