# -*- coding: utf-8 -*-
"""Unit tests for rolling Mean Absolute Deviation (MAD) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.statistics.mad import mad_ind, mad_numba, mad_polars


@pytest.mark.statistics
def test_mad_numba_basic(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test mad_numba on a random walk."""
    close = prices_random_walk
    length = 30
    result = mad_numba(close, length=length)

    assert result.shape == close.shape
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.statistics
def test_mad_numba_simple() -> None:
    """Test MAD on a simple array."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    length = 3
    result = mad_numba(prices, length=length)

    expected = np.full_like(prices, np.nan)
    for i in range(length - 1, len(prices)):
        window = prices[i - length + 1 : i + 1]
        expected[i] = np.mean(np.abs(window - np.mean(window)))
    assert_allclose(result[2:], expected[2:], rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_mad_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 30
    result_no_offset = mad_numba(close, length=length, offset=0)
    result_offset = mad_numba(close, length=length, offset=1, fillna=0.0)

    assert result_offset[0] == 0.0
    # fillna also replaces the warm-up NaNs of the shifted series
    no_offset_tail = result_no_offset[:-1]
    expected = np.where(np.isnan(no_offset_tail), 0.0, no_offset_tail)
    assert_allclose(result_offset[1:], expected, rtol=1e-6)


@pytest.mark.statistics
def test_mad_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test mad_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 30
    result = mad_ind(s, length=length)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(prices_random_walk),)
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.statistics
def test_mad_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test mad_polars adds a column correctly."""
    length = 30
    result_series = mad_polars(
        df_random_walk,
        close_col="close",
        length=length,
        output_col="MAD",
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == "MAD"
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk["close"].to_numpy()
    expected = mad_numba(close_arr, length=length)
    assert_allclose(
        result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.statistics
def test_mad_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    length = 3
    result_series = mad_polars(df, close_col="close", length=length)
    assert result_series.name == f"MAD_{length}"


@pytest.mark.statistics
def test_mad_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test mad_polars with offset and fillna."""
    length = 30
    result_series = mad_polars(
        df_random_walk,
        close_col="close",
        length=length,
        offset=1,
        fillna=0.0,
        output_col="MAD",
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk["close"].to_numpy()
    expected = mad_numba(close_arr, length=length, offset=1, fillna=0.0)
    assert_allclose(
        result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.statistics
def test_mad_numba_known_value() -> None:
    """Test the documented example: MAD of [1,2,3] windows is 2/3."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = mad_numba(prices, length=3)
    assert_allclose(result[2:], 2.0 / 3.0, rtol=1e-6)


@pytest.mark.statistics
def test_mad_numba_constant() -> None:
    """MAD of a constant series is exactly 0 after the warm-up."""
    length = 5
    prices = np.full(20, 5.0)
    result = mad_numba(prices, length=length)
    assert_allclose(result[length - 1 :], 0.0, atol=1e-12)


@pytest.mark.statistics
def test_mad_numba_nonfinite_window_is_nan() -> None:
    """NaN/inf in a window force NaN there; later windows recover.

    Regression test: the old incremental running sum `s` was poisoned by
    a single NaN/inf forever (mean stayed NaN for the rest of the array).
    """
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = mad_numba(close, length=3)
    # Windows [0..2], [1..3], [2..4] contain the NaN at index 2.
    assert np.isnan(result[2:5]).all()
    assert np.isfinite(result[5:]).all()
    # Windows [5..7], [6..8] -> values 5,6,7 / 6,7,8, MAD = 2/3.
    assert_allclose(result[5:], 2.0 / 3.0, rtol=1e-6)

    close_inf = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result_inf = mad_numba(close_inf, length=3)
    assert np.isnan(result_inf[2:4]).all()
    assert np.isfinite(result_inf[4:]).all()


@pytest.mark.statistics
def test_mad_numba_large_prices_accuracy() -> None:
    """Two-pass mean stays exact where incremental sums drift on big prices."""
    import math

    rng = np.random.default_rng(0)
    close = 100000.0 + rng.normal(0.0, 0.01, 120)
    length = 20
    result = mad_numba(close, length=length)

    for i in range(length - 1, len(close), 9):
        window = close[i - length + 1 : i + 1]
        mean = math.fsum(window) / length
        expected = math.fsum(abs(x - mean) for x in window) / length
        assert result[i] == pytest.approx(expected, rel=1e-9)


@pytest.mark.statistics
def test_mad_numba_length_too_short_raises() -> None:
    """Length < 2 is rejected."""
    prices = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="length must be >= 2"):
        mad_numba(prices, length=1)
    with pytest.raises(ValueError, match="length must be >= 2"):
        mad_numba(prices, length=0)


@pytest.mark.statistics
def test_mad_numba_length_exceeds_data() -> None:
    """If length > len(close), all outputs are NaN."""
    result = mad_numba(np.array([1.0, 2.0, 3.0]), length=5)
    assert result.shape == (3,)
    assert np.isnan(result).all()
