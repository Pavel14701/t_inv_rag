# -*- coding: utf-8 -*-
"""Unit tests for rolling Mean Absolute Deviation (MAD) module."""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...statistics.mad import mad_numba, mad_ind, mad_polars


@pytest.mark.statistics
def test_mad_numba_basic(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test mad_numba on a random walk."""
    close = prices_random_walk
    length = 30
    result = mad_numba(close, length=length)

    assert result.shape == close.shape
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1:]).all()


@pytest.mark.statistics
def test_mad_numba_simple() -> None:
    """Test MAD on a simple array."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    length = 3
    result = mad_numba(prices, length=length)

    expected = np.full_like(prices, np.nan)
    for i in range(length - 1, len(prices)):
        window = prices[i - length + 1:i + 1]
        expected[i] = np.mean(np.abs(window - np.mean(window)))
    assert_allclose(result[2:], expected[2:], rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_mad_numba_offset_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 30
    result_no_offset = mad_numba(close, length=length, offset=0)
    result_offset = mad_numba(close, length=length, offset=1, fillna=0.0)

    assert result_offset[0] == 0.0
    assert_allclose(result_offset[1:], result_no_offset[:-1], rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_mad_ind_with_pl_series(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test mad_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 30
    result = mad_ind(s, length=length)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(prices_random_walk),)
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1:]).all()


@pytest.mark.statistics
def test_mad_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test mad_polars adds a column correctly."""
    length = 30
    result_series = mad_polars(
        df_random_walk,
        close_col='close',
        length=length,
        output_col='MAD',
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == 'MAD'
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk['close'].to_numpy()
    expected = mad_numba(close_arr, length=length)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_mad_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    length = 3
    result_series = mad_polars(df, close_col='close', length=length)
    assert result_series.name == f'MAD_{length}'


@pytest.mark.statistics
def test_mad_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test mad_polars with offset and fillna."""
    length = 30
    result_series = mad_polars(
        df_random_walk,
        close_col='close',
        length=length,
        offset=1,
        fillna=0.0,
        output_col='MAD',
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk['close'].to_numpy()
    expected = mad_numba(close_arr, length=length, offset=1, fillna=0.0)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)