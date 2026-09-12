# -*- coding: utf-8 -*-
"""Unit and performance tests for Renko brick generation.

Tests cover:
- Numba-accelerated core function (_renko_nb)
- Universal renko() function with numpy and polars inputs
- renko_polars() DataFrame integration
- offset and fillna handling
- Edge cases (empty, single element, flat prices)
- Performance benchmarks for large arrays
"""

import time

import pytest
import numpy as np
import polars as pl

from ...candle.renko import _renko_nb, renko, renko_polars


RENKO_TEST_CASES = [
    # (prices, box_size, expected, description)
    (
        [100.0, 102.0, 104.0, 106.0, 108.0],
        2.0,
        [0, 1, 1, 1, 1],
        'uptrend one brick per step',
    ),
    (
        [100.0, 98.0, 96.0, 94.0, 92.0],
        2.0,
        [0, -1, -1, -1, -1],
        'downtrend one brick per step',
    ),
    (
        [100.0, 103.0, 101.0, 104.0, 98.0],
        2.0,
        [0, 1, 0, 1, -1],
        'mixed up/down movements',
    ),
    (
        [100.0, 100.5, 101.0, 99.5],
        2.0,
        [0, 0, 0, 0],
        'no movement within box size',
    ),
    (
        [100.0, 102.5, 104.0, 106.0],
        2.0,
        [0, 1, 1, 1],
        'partial up move (102.5 crosses 102)',
    ),
    (
        [100.0, 97.0, 95.0, 93.0],
        2.0,
        [0, -1, -1, -1],
        'partial down move (97.0 crosses 98)',
    ),
    (
        [100.0, 104.0, 96.0, 100.0],
        2.0,
        [0, 1, -1, 1],
        'up then down then back to anchor',
    ),
]


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize('prices, box_size, expected, desc', RENKO_TEST_CASES)
def test_renko_nb_parametrized(
    prices: list[float],
    box_size: float,
    expected: list[int],
    desc: str,
) -> None:
    """Test _renko_nb with various price series."""
    prices_arr = np.array(prices, dtype=np.float64)
    expected_arr = np.array(expected, dtype=np.int8)
    result = _renko_nb(prices_arr, box_size)
    np.testing.assert_array_equal(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize('prices, box_size, expected, desc', RENKO_TEST_CASES)
def test_renko_parametrized(
    prices: list[float],
    box_size: float,
    expected: list[int],
    desc: str,
) -> None:
    """Test renko() function with numpy array input."""
    prices_arr = np.array(prices, dtype=np.float64)
    expected_arr = np.array(expected, dtype=np.float64)
    result = renko(prices_arr, box_size)
    np.testing.assert_allclose(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize('prices, box_size, expected, desc', RENKO_TEST_CASES)
def test_renko_polars_series_parametrized(
    prices: list[float],
    box_size: float,
    expected: list[int],
    desc: str,
) -> None:
    """Test renko() with polars Series input."""
    s = pl.Series(prices)
    expected_arr = np.array(expected, dtype=np.float64)
    result = renko(s, box_size)
    np.testing.assert_allclose(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
def test_renko_on_uptrend(
    prices_uptrend: np.ndarray,
    box_size: float,
) -> None:
    """Test renko on a steady uptrend."""
    result = renko(prices_uptrend, box_size)
    assert len(result) == len(prices_uptrend)
    # All non-zero values should be +1 (up bricks)
    assert np.all((result == 0) | (result == 1))


@pytest.mark.unit
@pytest.mark.candle
def test_renko_on_downtrend(
    prices_downtrend: np.ndarray,
    box_size: float,
) -> None:
    """Test renko on a steady downtrend."""
    result = renko(prices_downtrend, box_size)
    assert len(result) == len(prices_downtrend)
    # All non-zero values should be -1 (down bricks)
    assert np.all((result == 0) | (result == -1))


@pytest.mark.unit
@pytest.mark.candle
def test_renko_on_sideways(
    prices_sideways: np.ndarray,
    box_size: float,
) -> None:
    """Test renko on a sideways market."""
    result = renko(prices_sideways, box_size)
    assert len(result) == len(prices_sideways)
    # There should be some zero values (no bricks)
    assert np.any(result == 0)


@pytest.mark.unit
@pytest.mark.candle
def test_renko_on_up_then_down(
    prices_up_then_down: np.ndarray,
    box_size: float,
) -> None:
    """Test renko on price series that rises then falls."""
    result = renko(prices_up_then_down, box_size)
    assert len(result) == len(prices_up_then_down)
    # There should be both positive and negative values
    assert np.any(result == 1)
    assert np.any(result == -1)


@pytest.mark.unit
@pytest.mark.candle
def test_renko_polars_df_uptrend(
    df_uptrend: pl.DataFrame,
    box_size: float,
) -> None:
    """Test renko_polars on a DataFrame with uptrend."""
    result_df = renko_polars(df_uptrend, price_col='close', box_size=box_size)
    assert 'RENKO' in result_df.columns
    assert len(result_df) == len(df_uptrend)
    # Check that the column is float64
    assert result_df['RENKO'].dtype == pl.Float64


@pytest.mark.unit
@pytest.mark.candle
def test_renko_polars_df_random_walk(
    df_random_walk: pl.DataFrame,
    box_size: float,
) -> None:
    """Test renko_polars on a random walk DataFrame."""
    result_df = renko_polars(
        df_random_walk,
        price_col='close',
        box_size=box_size
    )
    assert 'RENKO' in result_df.columns
    assert len(result_df) == len(df_random_walk)
    # There should be some non-zero values (some bricks)
    assert np.any(result_df['RENKO'].to_numpy() != 0)


@pytest.mark.unit
@pytest.mark.candle
def test_renko_nb_empty() -> None:
    """Test empty input array."""
    prices = np.array([], dtype=np.float64)
    result = _renko_nb(prices, 2.0)
    assert result.shape == (0,)
    assert result.dtype == np.int8


@pytest.mark.unit
@pytest.mark.candle
def test_renko_nb_single() -> None:
    """Test single element array."""
    prices = np.array([100.0], dtype=np.float64)
    result = _renko_nb(prices, 2.0)
    expected = np.array([0], dtype=np.int8)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@pytest.mark.candle
def test_renko_numpy_input() -> None:
    """Test renko() with numpy array input."""
    prices = np.array([100.0, 103.0, 101.0, 104.0, 98.0], dtype=np.float64)
    result = renko(prices, 2.0)
    expected = np.array([0.0, 1.0, 0.0, 1.0, -1.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


@pytest.mark.unit
@pytest.mark.candle
def test_renko_offset() -> None:
    """Test that offset shifts the output correctly."""
    prices = np.array([100.0, 102.0, 104.0, 106.0], dtype=np.float64)
    box_size = 2.0
    result_no_offset = renko(prices, box_size, offset=0)
    expected_no_offset = np.array([0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result_no_offset, expected_no_offset)

    # offset=1 -> shift forward and fill with 0
    result_offset = renko(prices, box_size, offset=1, fillna=0.0)
    assert len(result_offset) == len(prices)
    assert not np.isnan(result_offset).any()


@pytest.mark.unit
@pytest.mark.candle
def test_renko_fillna() -> None:
    """Test that fillna replaces NaN values."""
    prices = np.array([100.0, 102.0, 104.0, 106.0], dtype=np.float64)
    box_size = 2.0
    result = renko(prices, box_size, offset=1, fillna=0.0)
    assert not np.isnan(result).any()


@pytest.mark.unit
@pytest.mark.candle
def test_renko_polars() -> None:
    """Test renko_polars adds a column with correct values."""
    df = pl.DataFrame({'close': [100.0, 103.0, 101.0, 104.0, 98.0]})
    result_df = renko_polars(
        df,
        price_col='close',
        box_size=2.0,
        output_col='RENKO'
    )
    assert 'RENKO' in result_df.columns
    expected_values = [0.0, 1.0, 0.0, 1.0, -1.0]
    np.testing.assert_allclose(
        result_df['RENKO'].to_numpy(),
        np.array(expected_values)
    )


@pytest.mark.unit
@pytest.mark.candle
def test_renko_polars_with_offset_and_fillna() -> None:
    """Test renko_polars with offset and fillna parameters."""
    df = pl.DataFrame({'close': [100.0, 102.0, 104.0, 106.0]})
    result_df = renko_polars(
        df, price_col='close', box_size=2.0,
        offset=1, fillna=0.0, output_col='RENKO'
    )
    assert 'RENKO' in result_df.columns
    assert len(result_df) == len(df)
    assert not np.isnan(result_df['RENKO'].to_numpy()).any()


@pytest.mark.performance
@pytest.mark.candle
def test_renko_performance() -> None:
    """Performance benchmark for renko on large arrays."""
    sizes = [1000, 10000, 100000, 1000000]
    box_size = 2.0

    for size in sizes:
        prices = np.random.randn(size).astype(np.float64) * 10 + 100
        # warm-up
        _ = renko(prices, box_size)
        # measure
        start = time.perf_counter()
        result = renko(prices, box_size)
        elapsed = time.perf_counter() - start  # noqa: F841
        # sanity checks
        assert len(result) == size
        assert result.dtype == np.float64
        # Optional: print timing (comment out in CI)
        # print(f"Size {size}: {elapsed:.4f} sec")


@pytest.mark.performance
@pytest.mark.candle
def test_renko_nb_vs_python() -> None:
    """Compare Numba version against a pure-Python reference (if available)."""
    size = 100000
    prices = np.random.randn(size).astype(np.float64) * 10 + 100
    box_size = 2.0

    # Pure Python reference (slow) - just for illustration
    def renko_python(prices, box_size):
        n = len(prices)
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            return out
        anchor = prices[0]
        for i in range(1, n):
            p = prices[i]
            while p >= anchor + box_size:
                anchor += box_size
                out[i] = 1.0
            while p <= anchor - box_size:
                anchor -= box_size
                out[i] = -1.0
        return out
    # Measure Python
    start = time.perf_counter()
    _ = renko_python(prices, box_size)
    py_time = time.perf_counter() - start
    # Measure Numba (first call includes compilation, so we warm-up)
    _ = renko(prices[:1000], box_size)
    start = time.perf_counter()
    _ = renko(prices, box_size)
    nb_time = time.perf_counter() - start
    # Numba should be at least 2x faster on large arrays
    if py_time > 0.01 and nb_time > 0.0:
        assert nb_time < py_time * 0.5, (
            'Numba version should be faster'
        )
