# -*- coding: utf-8 -*-
"""Unit tests for stdev module (rolling standard deviation).

Tests cover:
- Numba core function (stdev_numba)
- ddof, offset, fillna
- Negative values
- stdev_ind backend selection
- stdev_polars and stdev_polars_multi integration
"""

import pytest
import numpy as np
import polars as pl
import numpy.typing as npt
from numpy.testing import assert_allclose

from ...statistics.stdev import (
    stdev_numba,
    stdev_talib,
    stdev_ind,
    stdev_polars,
    stdev_polars_multi,
)
from ...external import talib_available


@pytest.mark.statistics
def test_stdev_numba_basic(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test stdev_numba on a random walk."""
    close = prices_random_walk
    length = 3
    ddof = 1
    result = stdev_numba(close, length=length, ddof=ddof)

    # Manually compute std for each window
    n = len(close)
    expected = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = close[i - length + 1: i + 1]
        expected[i] = np.std(window, ddof=ddof)

    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_stdev_numba_ddof(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test that ddof affects the result correctly."""
    close = prices_random_walk
    length = 10
    result_ddof0 = stdev_numba(close, length=length, ddof=0)
    result_ddof1 = stdev_numba(close, length=length, ddof=1)

    valid_mask = ~np.isnan(result_ddof1)
    # Sample std (ddof=1) should be >= population std (ddof=0)
    assert np.all(result_ddof1[valid_mask] >= result_ddof0[valid_mask])


@pytest.mark.statistics
def test_stdev_numba_offset_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 3
    ddof = 1

    result_no_offset = stdev_numba(close, length=length, ddof=ddof, offset=0)
    result_offset = stdev_numba(close, length=length, ddof=ddof, offset=1, fillna=0.0)

    assert result_offset[0] == 0.0
    # fillna also replaces the warm-up NaNs of the shifted series
    no_offset_tail = result_no_offset[:-1]
    expected = np.where(np.isnan(no_offset_tail), 0.0, no_offset_tail)
    assert_allclose(result_offset[1:], expected, rtol=1e-6)


@pytest.mark.statistics
def test_stdev_numba_negative_values() -> None:
    """Test with negative values."""
    close = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)
    length = 3
    ddof = 1
    result = stdev_numba(close, length=length, ddof=ddof)

    expected = np.full_like(close, np.nan)
    for i in range(length - 1, len(close)):
        window = close[i - length + 1: i + 1]
        expected[i] = np.std(window, ddof=ddof)

    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# IEEE-754 corner-case tests
# -----------------------------------------------------------------------------

@pytest.mark.statistics
@pytest.mark.parametrize('algorithm', ['online', 'two_pass'])
def test_stdev_numba_nan_recovers(algorithm: str) -> None:
    """NaN poisons only the windows that contain it, then output recovers."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = stdev_numba(close, length=3, ddof=1, algorithm=algorithm)
    # Windows [0..2], [1..3], [2..4] all contain the NaN at index 2.
    assert np.isnan(result[2:5]).all()
    # Windows from index 5 on no longer contain it.
    assert np.isfinite(result[5:]).all()
    expected = np.std(close[5:8], ddof=1)
    assert_allclose(result[5], expected, rtol=1e-9)


@pytest.mark.statistics
@pytest.mark.parametrize('algorithm', ['online', 'two_pass'])
def test_stdev_numba_inf_is_nan_and_recovers(algorithm: str) -> None:
    """A +-inf value makes its windows NaN, later windows recover."""
    close = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = stdev_numba(close, length=3, ddof=1, algorithm=algorithm)
    assert np.isnan(result[2:4]).all()
    assert np.isfinite(result[4:]).all()


@pytest.mark.statistics
def test_stdev_numba_large_prices_twopass_accuracy() -> None:
    """Two-pass form stays exact where running sums of squares cancel."""
    rng = np.random.default_rng(0)
    close = 100000.0 + rng.normal(0.0, 0.01, 120)
    length = 20
    result = stdev_numba(close, length=length, ddof=1, algorithm='two_pass')

    for i in range(length - 1, len(close), 9):
        window = close[i - length + 1 : i + 1]
        expected = np.std(window, ddof=1)
        assert_allclose(result[i], expected, rtol=1e-9)


@pytest.mark.statistics
def test_stdev_numba_invalid_length_raises() -> None:
    """Passing length < 1 raises ValueError."""
    with pytest.raises(ValueError, match='length must be >= 1'):
        stdev_numba(np.array([1.0, 2.0, 3.0]), length=0)


@pytest.mark.statistics
def test_stdev_numba_invalid_ddof_raises() -> None:
    """Passing ddof outside [0, length) raises ValueError."""
    close = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='ddof must satisfy'):
        stdev_numba(close, length=3, ddof=3)
    with pytest.raises(ValueError, match='ddof must satisfy'):
        stdev_numba(close, length=3, ddof=-1)


@pytest.mark.statistics
def test_stdev_ind_uses_numba(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test stdev_ind uses Numba when TA-Lib is not requested."""
    close = prices_random_walk
    length = 3
    ddof = 1

    result_numba = stdev_ind(close, length=length, ddof=ddof, use_talib=False)
    expected = stdev_numba(close, length=length, ddof=ddof)
    assert_allclose(result_numba, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(
    not talib_available,
    reason='TA-Lib not installed',
)
@pytest.mark.statistics
def test_stdev_ind_uses_talib(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test stdev_ind uses TA-Lib when available and requested."""
    close = prices_random_walk
    length = 3

    result_talib = stdev_ind(close, length=length, use_talib=True)
    expected_talib = stdev_talib(close, length=length)
    assert_allclose(result_talib, expected_talib, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_stdev_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test stdev_polars adds a column correctly."""
    length = 30
    result_series = stdev_polars(
        df_random_walk,
        close_col='close',
        length=length,
        use_talib=False,
        output_col='STDEV'
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == 'STDEV'
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk['close'].to_numpy()
    expected = stdev_numba(close_arr, length=length, ddof=1)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_stdev_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    length = 3
    result_series = stdev_polars(df, close_col='close', length=length)
    assert result_series.name == f'STDEV_{length}'


@pytest.mark.statistics
def test_stdev_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test stdev_polars with offset and fillna."""
    length = 30
    result_series = stdev_polars(
        df_random_walk,
        close_col='close',
        length=length,
        offset=1,
        fillna=0.0,
        use_talib=False,
        output_col='STDEV'
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk['close'].to_numpy()
    expected = stdev_numba(close_arr, length=length, ddof=1, offset=1, fillna=0.0)
    assert_allclose(result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.statistics
def test_stdev_polars_multi_basic(df_ohlc: pl.DataFrame) -> None:
    """Test stdev_polars_multi adds columns for multiple columns."""
    columns = ['open', 'high', 'low', 'close']
    length = 30
    result_df = stdev_polars_multi(
        df_ohlc,
        columns=columns,
        length=length,
        use_talib=False,
        suffix='_stdev'
    )

    for col in columns:
        new_col = f'{col}_stdev'
        assert new_col in result_df.columns
        assert result_df[new_col].dtype == pl.Float64

        arr = df_ohlc[col].to_numpy()
        expected = stdev_numba(arr, length=length, ddof=1)
        assert_allclose(
            result_df[new_col].to_numpy(),
            expected,
            rtol=1e-6,
            equal_nan=True
        )


@pytest.mark.statistics
def test_stdev_polars_multi_offset_fillna(df_ohlc: pl.DataFrame) -> None:
    """Test stdev_polars_multi with offset and fillna."""
    columns = ['open', 'close']
    length = 30
    result_df = stdev_polars_multi(
        df_ohlc,
        columns=columns,
        length=length,
        offset=1,
        fillna=0.0,
        use_talib=False,
        suffix='_stdev'
    )

    for col in columns:
        assert result_df[f'{col}_stdev'][0] == 0.0

    close_arr = df_ohlc['close'].to_numpy()
    expected = stdev_numba(close_arr, length=length, ddof=1, offset=1, fillna=0.0)
    assert_allclose(
        result_df['close_stdev'].to_numpy(),
        expected,
        rtol=1e-6,
        equal_nan=True
    )


@pytest.mark.statistics
def test_stdev_polars_multi_empty_columns() -> None:
    """Test stdev_polars_multi with empty list of columns."""
    df = pl.DataFrame({'a': [1.0, 2.0, 3.0]})
    result_df = stdev_polars_multi(df, columns=[], length=3)
    assert result_df.columns == df.columns
    assert len(result_df) == len(df)