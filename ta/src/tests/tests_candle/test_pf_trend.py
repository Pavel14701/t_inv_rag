# -*- coding: utf-8 -*-
"""Unit and performance tests for Point & Figure trend state (X/O columns).

Tests cover:
- Numba-accelerated core function (_pf_trend_nb)
- Universal pf_trend() function with numpy and polars inputs
- pf_trend_polars() DataFrame integration
- offset and fillna handling
- Edge cases (empty, single element, flat prices)
- Performance benchmarks for large arrays
"""

import time

import pytest
import numpy as np
import polars as pl

from ...candle.pf_trend import _pf_trend_nb, pf_trend, pf_trend_polars


PF_TREND_TEST_CASES = [
    # (prices, box_size, reversal, expected, description)
    (
        [100.0, 102.0, 104.0, 106.0, 108.0],
        2.0,
        3,
        [0, 1, 1, 1, 1],
        'uptrend single column',
    ),
    (
        [100.0, 98.0, 96.0, 94.0, 92.0],
        2.0,
        3,
        [0, -1, -1, -1, -1],
        'downtrend single column',
    ),
    (
        [100.0, 103.0, 101.0, 98.0, 95.0],
        2.0,
        3,
        [0, 1, 1, 1, -1],
        'uptrend then reversal (3 boxes down)',
    ),
    (
        [100.0, 97.0, 99.0, 102.0, 105.0],
        2.0,
        3,
        [0, -1, -1, 1, 1],
        'downtrend then reversal (3 boxes up)',
    ),
    (
        [100.0, 101.0, 102.0, 103.0, 104.0],
        2.0,
        3,
        [0, 0, 1, 1, 1],
        'no movement within box_size',
    ),
    (
        [100.0, 106.0, 98.0, 92.0, 86.0],
        2.0,
        3,
        [0, 1, -1, -1, -1],
        'up then down with multiple boxes',
    ),
    (
        [100.0, 94.0, 106.0, 112.0, 118.0],
        2.0,
        3,
        [0, -1, 1, 1, 1],
        'down then up with multiple boxes',
    ),
    (
        [100.0, 105.0, 102.0, 99.0, 96.0],
        2.0,
        3,
        [0, 1, 1, 1, -1],
        'up then down with reversal exactly at threshold',
    ),
]


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize(
    'prices, box_size, reversal, expected, desc',
    PF_TREND_TEST_CASES
)
def test_pf_trend_nb_parametrized(
    prices: list[float],
    box_size: float,
    reversal: int,
    expected: list[int],
    desc: str,
) -> None:
    """Test _pf_trend_nb with various price series."""
    prices_arr = np.array(prices, dtype=np.float64)
    expected_arr = np.array(expected, dtype=np.int8)
    result = _pf_trend_nb(prices_arr, box_size, reversal)
    np.testing.assert_array_equal(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize(
    'prices, box_size, reversal, expected, desc',
    PF_TREND_TEST_CASES
)
def test_pf_trend_parametrized(
    prices: list[float],
    box_size: float,
    reversal: int,
    expected: list[int],
    desc: str,
) -> None:
    """Test pf_trend() function with numpy array input."""
    prices_arr = np.array(prices, dtype=np.float64)
    expected_arr = np.array(expected, dtype=np.float64)
    result = pf_trend(prices_arr, box_size, reversal)
    np.testing.assert_allclose(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize(
    'prices, box_size, reversal, expected, desc',
    PF_TREND_TEST_CASES
)
def test_pf_trend_polars_series_parametrized(
    prices: list[float],
    box_size: float,
    reversal: int,
    expected: list[int],
    desc: str,
) -> None:
    """Test pf_trend() with polars Series input."""
    s = pl.Series(prices)
    expected_arr = np.array(expected, dtype=np.float64)
    result = pf_trend(s, box_size, reversal)
    np.testing.assert_allclose(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_on_uptrend(
    prices_uptrend: np.ndarray,
    box_size: float,
    reversal: int,
) -> None:
    """Test pf_trend on a steady uptrend."""
    result = pf_trend(prices_uptrend, box_size, reversal)
    assert len(result) == len(prices_uptrend)
    # All non-zero values should be +1 (up columns)
    assert np.all((result == 0) | (result == 1))


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_on_downtrend(
    prices_downtrend: np.ndarray,
    box_size: float,
    reversal: int,
) -> None:
    """Test pf_trend on a steady downtrend."""
    result = pf_trend(prices_downtrend, box_size, reversal)
    assert len(result) == len(prices_downtrend)
    # All non-zero values should be -1 (down columns)
    assert np.all((result == 0) | (result == -1))


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_on_sideways(
    prices_sideways: np.ndarray,
    box_size: float,
    reversal: int,
) -> None:
    """Test pf_trend on a sideways market."""
    result = pf_trend(prices_sideways, box_size, reversal)
    assert len(result) == len(prices_sideways)
    # There should be some zero values (no clear trend)
    assert np.any(result == 0)


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_on_up_then_down(
    prices_up_then_down: np.ndarray,
    box_size: float,
    reversal: int,
) -> None:
    """Test pf_trend on price series that rises then falls."""
    result = pf_trend(prices_up_then_down, box_size, reversal)
    assert len(result) == len(prices_up_then_down)
    # There should be both positive and negative values
    assert np.any(result == 1)
    assert np.any(result == -1)


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_polars_df_uptrend(
    df_uptrend: pl.DataFrame,
    box_size: float,
    reversal: int,
) -> None:
    """Test pf_trend_polars on a DataFrame with uptrend."""
    result_df = pf_trend_polars(
        df_uptrend,
        price_col='close',
        box_size=box_size,
        reversal=reversal
    )
    assert 'PF_TREND' in result_df.columns
    assert len(result_df) == len(df_uptrend)
    # Check that the column is float64
    assert result_df['PF_TREND'].dtype == pl.Float64


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_polars_df_random_walk(
    df_random_walk: pl.DataFrame,
    box_size: float,
    reversal: int,
) -> None:
    """Test pf_trend_polars on a random walk DataFrame."""
    result_df = pf_trend_polars(
        df_random_walk,
        price_col='close',
        box_size=box_size,
        reversal=reversal
    )
    assert 'PF_TREND' in result_df.columns
    assert len(result_df) == len(df_random_walk)
    # There should be some non-zero values (some columns)
    assert np.any(result_df['PF_TREND'].to_numpy() != 0)


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_nb_empty() -> None:
    """Test empty input array."""
    prices = np.array([], dtype=np.float64)
    result = _pf_trend_nb(prices, 2.0, 3)
    assert result.shape == (0,)
    assert result.dtype == np.int8


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_nb_single() -> None:
    """Test single element array."""
    prices = np.array([100.0], dtype=np.float64)
    result = _pf_trend_nb(prices, 2.0, 3)
    expected = np.array([0], dtype=np.int8)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_numpy_input() -> None:
    """Test pf_trend() with numpy array input."""
    prices = np.array([100.0, 103.0, 106.0, 100.0, 94.0], dtype=np.float64)
    result = pf_trend(prices, 2.0, 3)
    expected = np.array([0.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_offset() -> None:
    """Test that offset shifts the output correctly."""
    prices = np.array([100.0, 102.0, 104.0, 106.0], dtype=np.float64)
    box_size = 2.0
    reversal = 3
    # offset=0 (default) -> [0, 1, 1, 1]
    result_no_offset = pf_trend(prices, box_size, reversal, offset=0)
    expected_no_offset = np.array([0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result_no_offset, expected_no_offset)
    # offset=1 -> shift forward and fill with 0
    result_offset = pf_trend(prices, box_size, reversal, offset=1, fillna=0.0)
    assert len(result_offset) == len(prices)
    assert not np.isnan(result_offset).any()


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_fillna() -> None:
    """Test that fillna replaces NaN values."""
    prices = np.array([100.0, 102.0, 104.0, 106.0], dtype=np.float64)
    box_size = 2.0
    reversal = 3
    result = pf_trend(prices, box_size, reversal, offset=1, fillna=0.0)
    assert not np.isnan(result).any()


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_polars() -> None:
    """Test pf_trend_polars adds a column with correct values."""
    df = pl.DataFrame({'close': [100.0, 103.0, 106.0, 100.0, 94.0]})
    result_df = pf_trend_polars(
        df, price_col='close',
        box_size=2.0, reversal=3,
        output_col='PF_TREND'
    )
    assert 'PF_TREND' in result_df.columns
    expected_values = [0.0, 1.0, 1.0, -1.0, -1.0]
    np.testing.assert_allclose(
        result_df['PF_TREND'].to_numpy(),
        np.array(expected_values)
    )


@pytest.mark.unit
@pytest.mark.candle
def test_pf_trend_polars_with_offset_and_fillna() -> None:
    """Test pf_trend_polars with offset and fillna parameters."""
    df = pl.DataFrame({'close': [100.0, 102.0, 104.0, 106.0]})
    result_df = pf_trend_polars(
        df, price_col='close', box_size=2.0,
        reversal=3, offset=1, fillna=0.0,
        output_col='PF_TREND'
    )
    assert 'PF_TREND' in result_df.columns
    assert len(result_df) == len(df)
    assert not np.isnan(result_df['PF_TREND'].to_numpy()).any()


@pytest.mark.performance
@pytest.mark.candle
def test_pf_trend_performance() -> None:
    """Performance benchmark for pf_trend on large arrays."""
    sizes = [1000, 10000, 100000, 1000000]
    box_size = 2.0
    reversal = 3
    for size in sizes:
        prices = np.random.randn(size).astype(np.float64) * 10 + 100
        # warm-up
        _ = pf_trend(prices, box_size, reversal)
        # measure
        start = time.perf_counter()
        result = pf_trend(prices, box_size, reversal)
        elapsed = time.perf_counter() - start  # noqa: F841
        # sanity checks
        assert len(result) == size
        assert result.dtype == np.float64
        # Optional: print timing (comment out in CI)
        # print(f"Size {size}: {elapsed:.4f} sec")


# Pure Python reference (slow) - just for illustration
def pf_trend_python(prices, box_size, reversal):  # noqa: D103, C901
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    p0 = prices[0]
    cur_kind = 0
    col_top = p0
    col_bottom = p0
    for i in range(1, n):
        p = prices[i]
        if cur_kind == 0:
            if p >= p0 + box_size:
                cur_kind = 1
                col_bottom = np.floor(p0 / box_size) * box_size
                col_top = np.floor(p / box_size) * box_size
            elif p <= p0 - box_size:
                cur_kind = -1
                col_top = np.floor(p0 / box_size) * box_size
                col_bottom = np.floor(p / box_size) * box_size
            out[i] = cur_kind
            continue
        if cur_kind == 1:
            needed_up = col_top + box_size
            if p >= needed_up:
                col_top = np.floor(p / box_size) * box_size
                out[i] = 1
                continue
            rev_level = col_top - box_size * reversal
            if p <= rev_level:
                cur_kind = -1
                col_bottom = np.floor(p / box_size) * box_size
                out[i] = -1
                continue
            out[i] = 1
        else:
            needed_down = col_bottom - box_size
            if p <= needed_down:
                col_bottom = np.floor(p / box_size) * box_size
                out[i] = -1
                continue
            rev_level = col_bottom + box_size * reversal
            if p >= rev_level:
                cur_kind = 1
                col_top = np.floor(p / box_size) * box_size
                out[i] = 1
                continue
            out[i] = -1
    return out


@pytest.mark.performance
@pytest.mark.candle
def test_pf_trend_nb_vs_python() -> None:
    """Compare Numba version against a pure-Python reference (if available)."""
    size = 100000
    prices = np.random.randn(size).astype(np.float64) * 10 + 100
    box_size = 2.0
    reversal = 3
    # Measure Python
    start = time.perf_counter()
    _ = pf_trend_python(prices, box_size, reversal)
    py_time = time.perf_counter() - start
    # Measure Numba (first call includes compilation, so we warm-up)
    _ = pf_trend(prices[:1000], box_size, reversal)
    start = time.perf_counter()
    _ = pf_trend(prices, box_size, reversal)
    nb_time = time.perf_counter() - start
    # Numba should be at least 2x faster on large arrays
    if py_time > 0.01 and nb_time > 0.0:
        assert nb_time < py_time * 0.5, 'Numba version should be faster'
