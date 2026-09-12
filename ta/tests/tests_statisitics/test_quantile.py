# -*- coding: utf-8 -*-
"""Unit tests for rolling quantile (QTL) module."""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...statistics.quantile import quantile_numba, quantile_ind, quantile_polars


@pytest.mark.statistics
def test_quantile_numba_basic(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test quantile_numba with q=0.5 (median)."""
    close = prices_random_walk
    length = 30
    result = quantile_numba(close, length=length, q=0.5)

    assert result.shape == close.shape
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1:]).all()


@pytest.mark.statistics
def test_quantile_numba_q_values() -> None:
    """Test quantile at different q values on a simple sorted series."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    length = 5
    result_q25 = quantile_numba(prices, length=length, q=0.25)
    result_q50 = quantile_numba(prices, length=length, q=0.5)
    result_q75 = quantile_numba(prices, length=length, q=0.75)

    # Last window: [6,7,8,9,10]
    # index = round(q * (length-1))
    # q=0.25 -> round(1) = 1 -> value 7
    # q=0.5  -> round(2) = 2 -> value 8
    # q=0.75 -> round(3) = 3 -> value 9
    assert_allclose(result_q25[-1], 7.0, rtol=1e-6)
    assert_allclose(result_q50[-1], 8.0, rtol=1e-6)
    assert_allclose(result_q75[-1], 9.0, rtol=1e-6)


@pytest.mark.statistics
def test_quantile_numba_offset_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 30
    result_no_offset = quantile_numba(close, length=length, q=0.5, offset=0)
    result_offset = quantile_numba(close, length=length, q=0.5, offset=1, fillna=0.0)

    assert result_offset[0] == 0.0
    # fillna also replaces the warm-up NaNs of the shifted series
    no_offset_tail = result_no_offset[:-1]
    expected = np.where(np.isnan(no_offset_tail), 0.0, no_offset_tail)
    assert_allclose(result_offset[1:], expected, rtol=1e-6)


@pytest.mark.statistics
def test_quantile_ind_with_pl_series(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test quantile_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 30
    result = quantile_ind(s, length=length, q=0.5)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(prices_random_walk),)
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1:]).all()


@pytest.mark.statistics
def test_quantile_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test quantile_polars adds a column correctly."""
    length = 30
    q = 0.5
    result_series = quantile_polars(
        df_random_walk,
        close_col='close',
        length=length,
        q=q,
        output_col='QTL',
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == 'QTL'
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk['close'].to_numpy()
    expected = quantile_numba(close_arr, length=length, q=q)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_quantile_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    length = 3
    q = 0.5
    result_series = quantile_polars(df, close_col='close', length=length, q=q)
    assert result_series.name == f'QTL_{length}_{q}'


@pytest.mark.statistics
def test_quantile_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test quantile_polars with offset and fillna."""
    length = 30
    q = 0.5
    result_series = quantile_polars(
        df_random_walk,
        close_col='close',
        length=length,
        q=q,
        offset=1,
        fillna=0.0,
        output_col='QTL',
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk['close'].to_numpy()
    expected = quantile_numba(close_arr, length=length, q=q, offset=1, fillna=0.0)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# IEEE-754 corner-case tests
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_quantile_numba_numpy_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Nearest-rank quantile matches np.quantile(method='nearest')."""
    close = prices_random_walk
    length = 20
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = quantile_numba(close, length=length, q=q)
        for i in range(length - 1, len(close), 11):
            window = close[i - length + 1 : i + 1]
            expected = np.quantile(window, q, method='nearest')
            assert_allclose(result[i], expected, rtol=1e-12)


@pytest.mark.statistics
def test_quantile_numba_nan_recovers() -> None:
    """NaN poisons only the windows that contain it, then output recovers."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = quantile_numba(close, length=3, q=0.5)
    # Without the finite check, sort() would push the NaN to the end and
    # silently return 4.0 for the window [2, 4, nan].
    assert np.isnan(result[2:5]).all()
    assert result[5] == pytest.approx(5.0)  # median of [4, 5, 6]


@pytest.mark.statistics
def test_quantile_numba_inf_is_nan_and_recovers() -> None:
    """A +-inf value makes its windows NaN, later windows recover."""
    close = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = quantile_numba(close, length=3, q=0.0)
    assert np.isnan(result[2:4]).all()
    assert result[4] == pytest.approx(3.0)  # min of [3, 4, 5]


@pytest.mark.statistics
def test_quantile_numba_invalid_length_raises() -> None:
    """Passing length < 1 raises ValueError."""
    with pytest.raises(ValueError, match='length must be >= 1'):
        quantile_numba(np.array([1.0, 2.0, 3.0]), length=0)


@pytest.mark.statistics
def test_quantile_numba_invalid_q_raises() -> None:
    """Passing q outside [0, 1] raises ValueError."""
    prices = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='q must be between 0 and 1'):
        quantile_numba(prices, length=2, q=-0.1)
    with pytest.raises(ValueError, match='q must be between 0 and 1'):
        quantile_numba(prices, length=2, q=1.1)