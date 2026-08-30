# -*- coding: utf-8 -*-
"""Unit tests for rolling skewness (SKEW) module."""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...statistics.skew import (
    _skew_numba_core,
    skew_numba,
    skew_ind,
    skew_polars,
)


def test_skew_numba_core_basic() -> None:
    """Test _skew_numba_core on a simple symmetric series."""
    prices = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    length = 4
    result = _skew_numba_core(prices, length)
    expected = np.full_like(prices, np.nan)
    expected[-1] = 0.0
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


def test_skew_numba_core_positive() -> None:
    """Test positive skewness (right tail)."""
    prices = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float64)
    length = 4
    result = _skew_numba_core(prices, length)
    assert result[-1] > 0


def test_skew_numba_core_negative() -> None:
    """Test negative skewness (left tail)."""
    # Series with left tail: low value 1, then 100,101,102
    prices = np.array([1.0, 100.0, 101.0, 102.0], dtype=np.float64)
    length = 4
    result = _skew_numba_core(prices, length)
    assert result[-1] < 0


@pytest.mark.statistics
def test_skew_numba_basic(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test skew_numba on a random walk."""
    close = prices_random_walk
    length = 30
    result = skew_numba(close, length=length)

    assert result.shape == close.shape
    assert result.dtype == np.float64
    assert np.isnan(result[:length - 1]).all()
    assert np.isfinite(result[length - 1:]).all()


@pytest.mark.statistics
def test_skew_numba_offset_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 30
    result_no_offset = skew_numba(close, length=length, offset=0)
    result_offset = skew_numba(close, length=length, offset=1, fillna=0.0)

    assert result_offset[0] == 0.0
    assert_allclose(result_offset[1:], result_no_offset[:-1], rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_skew_ind_with_pl_series(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test skew_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 30
    result = skew_ind(s, length=length)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(prices_random_walk),)
    assert result.dtype == np.float64
    assert np.isnan(result[:length - 1]).all()
    assert np.isfinite(result[length - 1:]).all()


@pytest.mark.statistics
def test_skew_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test skew_polars adds a column correctly."""
    length = 30
    result_series = skew_polars(
        df_random_walk,
        close_col='close',
        length=length,
        output_col='SKEW',
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == 'SKEW'
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk['close'].to_numpy()
    expected = skew_numba(close_arr, length=length)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_skew_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    length = 3
    result_series = skew_polars(df, close_col='close', length=length)
    assert result_series.name == f'SKEW_{length}'


@pytest.mark.statistics
def test_skew_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test skew_polars with offset and fillna."""
    length = 30
    result_series = skew_polars(
        df_random_walk,
        close_col='close',
        length=length,
        offset=1,
        fillna=0.0,
        output_col='SKEW',
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk['close'].to_numpy()
    expected = skew_numba(close_arr, length=length, offset=1, fillna=0.0)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)