# -*- coding: utf-8 -*-
"""Unit tests for variance module (rolling variance)."""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...statistics.variance import (
    variance_numba,
    variance_talib,
    variance_ind,
    variance_polars,
)
from ...external import talib_available


# -----------------------------------------------------------------------------
# Numba core tests
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_variance_numba_basic(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test variance_numba on a random walk."""
    close = prices_random_walk
    length = 3
    ddof = 1
    result = variance_numba(close, length=length, ddof=ddof)

    n = len(close)
    expected = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = close[i - length + 1 : i + 1]
        expected[i] = np.var(window, ddof=ddof)

    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_variance_numba_ddof(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test that ddof affects the result correctly."""
    close = prices_random_walk
    length = 10
    result_ddof0 = variance_numba(close, length=length, ddof=0)
    result_ddof1 = variance_numba(close, length=length, ddof=1)

    valid_mask = ~np.isnan(result_ddof1)
    # Population variance (ddof=0) should be <= sample variance (ddof=1)
    assert np.all(result_ddof0[valid_mask] <= result_ddof1[valid_mask])


@pytest.mark.statistics
def test_variance_numba_offset_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 3
    ddof = 1

    result_no_offset = variance_numba(close, length=length, ddof=ddof, offset=0)
    result_offset = variance_numba(
        close,
        length=length,
        ddof=ddof,
        offset=1,
        fillna=0.0,
    )

    assert result_offset[0] == 0.0
    assert_allclose(result_offset[1:], result_no_offset[:-1], rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Backend selection tests
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_variance_ind_uses_numba(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test variance_ind uses Numba when TA-Lib is not requested."""
    close = prices_random_walk
    length = 3
    ddof = 1

    result_numba = variance_ind(close, length=length, ddof=ddof, use_talib=False)
    expected = variance_numba(close, length=length, ddof=ddof)
    assert_allclose(result_numba, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(
    not talib_available,
    reason='TA-Lib not installed',
)
@pytest.mark.statistics
def test_variance_ind_uses_talib(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test variance_ind uses TA-Lib when available and requested."""
    close = prices_random_walk
    length = 3

    result_talib = variance_ind(close, length=length, use_talib=True)
    expected_talib = variance_talib(close, length=length)
    assert_allclose(result_talib, expected_talib, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_variance_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test variance_polars adds a column correctly."""
    length = 30
    result_series = variance_polars(
        df_random_walk,
        close_col='close',
        length=length,
        use_talib=False,
        output_col='VAR',
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == 'VAR'
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk['close'].to_numpy()
    expected = variance_numba(close_arr, length=length, ddof=1)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_variance_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    length = 3
    result_series = variance_polars(df, close_col='close', length=length, use_talib=False)
    assert result_series.name == f'VAR_{length}'


@pytest.mark.statistics
def test_variance_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test variance_polars with offset and fillna."""
    length = 30
    result_series = variance_polars(
        df_random_walk,
        close_col='close',
        length=length,
        offset=1,
        fillna=0.0,
        use_talib=False,
        output_col='VAR',
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk['close'].to_numpy()
    expected = variance_numba(
        close_arr,
        length=length,
        ddof=1,
        offset=1,
        fillna=0.0,
    )
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_variance_ind_with_pl_series(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test variance_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 3
    ddof = 1

    result = variance_ind(s, length=length, ddof=ddof, use_talib=False)
    expected = variance_numba(prices_random_walk, length=length, ddof=ddof)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)