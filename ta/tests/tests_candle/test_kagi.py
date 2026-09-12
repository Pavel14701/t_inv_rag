# -*- coding: utf-8 -*-
"""Unit and performance tests for Kagi line (yin/yang) generation.

Tests cover:
- Numba-accelerated core function (_kagi_nb)
- Universal kagi() function with numpy and polars inputs
- kagi_polars() DataFrame integration
- offset and fillna handling
- Edge cases (empty, single element, flat prices)
- Performance benchmarks for large arrays
"""

import time

import pytest
import numpy as np
import polars as pl

from ...candle.kagi import _kagi_nb, kagi, kagi_polars


# -----------------------------------------------------------------------------
# Parameterized test data (exact algorithm validation)
# -----------------------------------------------------------------------------

KAGI_TEST_CASES = [
    # (prices, reversal, expected, description)
    (
        [100.0, 102.0, 104.0, 106.0, 108.0],
        2.0,
        [0, 1, 1, 1, 1],
        'uptrend: new highs',
    ),
    (
        [100.0, 98.0, 96.0, 94.0, 92.0],
        2.0,
        [0, -1, -1, -1, -1],
        'downtrend: new lows',
    ),
    (
        [100.0, 103.0, 100.0, 98.0, 96.0],
        2.0,
        [0, 1, -1, -1, -1],
        'up then reversal (2 points down)',
    ),
    (
        [100.0, 97.0, 100.0, 102.0, 104.0],
        2.0,
        [0, -1, 1, 1, 1],
        'down then reversal (2 points up)',
    ),
    (
        [100.0, 101.0, 101.5, 101.0, 100.5],
        2.0,
        [0, 0, 0, 0, 0],
        'no movement within reversal threshold (all below 102)',
    ),
    (
        [100.0, 105.0, 104.0, 99.0, 96.0],
        2.0,
        [0, 1, 1, -1, -1],
        'up then down with multiple points (no immediate reversal)',
    ),
    (
        [100.0, 95.0, 94.0, 101.0, 104.0],
        2.0,
        [0, -1, -1, 1, 1],
        'down then up with multiple points',
    ),
    (
        [100.0, 103.0, 101.5, 99.5, 98.0],
        2.0,
        [0, 1, 1, -1, -1],
        'up then reversal exactly at threshold (avoid immediate reversal)',
    ),
]


# -----------------------------------------------------------------------------
# Parameterized tests for exact algorithm validation
# -----------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize('prices, reversal, expected, desc', KAGI_TEST_CASES)
def test_kagi_nb_parametrized(
    prices: list[float],
    reversal: float,
    expected: list[int],
    desc: str,
) -> None:
    """Test _kagi_nb with various price series."""
    prices_arr = np.array(prices, dtype=np.float64)
    expected_arr = np.array(expected, dtype=np.int8)
    result = _kagi_nb(prices_arr, reversal)
    np.testing.assert_array_equal(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize(
    'prices, reversal, expected, desc',
    KAGI_TEST_CASES
)
def test_kagi_parametrized(
    prices: list[float],
    reversal: float,
    expected: list[int],
    desc: str,
) -> None:
    """Test kagi() function with numpy array input."""
    prices_arr = np.array(prices, dtype=np.float64)
    expected_arr = np.array(expected, dtype=np.float64)
    result = kagi(prices_arr, reversal)
    np.testing.assert_allclose(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


@pytest.mark.unit
@pytest.mark.candle
@pytest.mark.parametrize('prices, reversal, expected, desc', KAGI_TEST_CASES)
def test_kagi_polars_series_parametrized(
    prices: list[float],
    reversal: float,
    expected: list[int],
    desc: str,
) -> None:
    """Test kagi() with polars Series input."""
    s = pl.Series(prices)
    expected_arr = np.array(expected, dtype=np.float64)
    result = kagi(s, reversal)
    np.testing.assert_allclose(
        result,
        expected_arr,
        err_msg=f'Failed for: {desc}'
    )


# -----------------------------------------------------------------------------
# Tests using shared fixtures for realistic price patterns
# -----------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.candle
def test_kagi_on_uptrend(
    prices_uptrend: np.ndarray,
    reversal: float,
) -> None:
    """Test kagi on a steady uptrend."""
    result = kagi(prices_uptrend, reversal)
    assert len(result) == len(prices_uptrend)
    # All non-zero values should be +1 (yang)
    assert np.all((result == 0) | (result == 1))


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_on_downtrend(
    prices_downtrend: np.ndarray,
    reversal: float,
) -> None:
    """Test kagi on a steady downtrend."""
    result = kagi(prices_downtrend, reversal)
    assert len(result) == len(prices_downtrend)
    # All non-zero values should be -1 (yin)
    assert np.all((result == 0) | (result == -1))


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_on_sideways(
    prices_sideways: np.ndarray,
    reversal: float,
) -> None:
    """Test kagi on a sideways market."""
    result = kagi(prices_sideways, reversal)
    assert len(result) == len(prices_sideways)
    # There should be some zero values (no clear direction)
    assert np.any(result == 0)


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_on_up_then_down(
    prices_up_then_down: np.ndarray,
    reversal: float,
) -> None:
    """Test kagi on price series that rises then falls."""
    result = kagi(prices_up_then_down, reversal)
    assert len(result) == len(prices_up_then_down)
    # There should be both positive and negative values
    assert np.any(result == 1)
    assert np.any(result == -1)


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_polars_df_uptrend(
    df_uptrend: pl.DataFrame,
    reversal: float,
) -> None:
    """Test kagi_polars on a DataFrame with uptrend."""
    result_df = kagi_polars(df_uptrend, price_col='close', reversal=reversal)
    assert 'KAGI' in result_df.columns
    assert len(result_df) == len(df_uptrend)
    # Check that the column is float64
    assert result_df['KAGI'].dtype == pl.Float64


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_polars_df_random_walk(
    df_random_walk: pl.DataFrame,
    reversal: float,
) -> None:
    """Test kagi_polars on a random walk DataFrame."""
    result_df = kagi_polars(
        df_random_walk,
        price_col='close',
        reversal=reversal
    )
    assert 'KAGI' in result_df.columns
    assert len(result_df) == len(df_random_walk)
    # There should be some non-zero values (some lines)
    assert np.any(result_df['KAGI'].to_numpy() != 0)


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_nb_empty() -> None:
    """Test empty input array."""
    prices = np.array([], dtype=np.float64)
    result = _kagi_nb(prices, 2.0)
    assert result.shape == (0,)
    assert result.dtype == np.int8


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_nb_single() -> None:
    """Test single element array."""
    prices = np.array([100.0], dtype=np.float64)
    result = _kagi_nb(prices, 2.0)
    expected = np.array([0], dtype=np.int8)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_numpy_input() -> None:
    """Test kagi() with numpy array input."""
    prices = np.array([100.0, 105.0, 104.0, 99.0, 96.0], dtype=np.float64)
    result = kagi(prices, 2.0)
    expected = np.array([0.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_offset() -> None:
    """Test that offset shifts the output correctly."""
    prices = np.array([100.0, 102.0, 104.0, 106.0], dtype=np.float64)
    reversal = 2.0
    # offset=0 (default) -> [0, 1, 1, 1]
    result_no_offset = kagi(prices, reversal, offset=0)
    expected_no_offset = np.array([0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result_no_offset, expected_no_offset)
    # offset=1 -> shift forward and fill with 0
    result_offset = kagi(prices, reversal, offset=1, fillna=0.0)
    assert len(result_offset) == len(prices)
    assert not np.isnan(result_offset).any()


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_fillna() -> None:
    """Test that fillna replaces NaN values."""
    prices = np.array([100.0, 102.0, 104.0, 106.0], dtype=np.float64)
    reversal = 2.0
    result = kagi(prices, reversal, offset=1, fillna=0.0)
    assert not np.isnan(result).any()


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_polars() -> None:
    """Test kagi_polars adds a column with correct values."""
    df = pl.DataFrame({'close': [100.0, 105.0, 104.0, 99.0, 96.0]})
    result_df = kagi_polars(
        df, price_col='close',
        reversal=2.0, output_col='KAGI'
    )
    assert 'KAGI' in result_df.columns
    expected_values = [0.0, 1.0, 1.0, -1.0, -1.0]
    np.testing.assert_allclose(
        result_df['KAGI'].to_numpy(),
        np.array(expected_values)
    )


@pytest.mark.unit
@pytest.mark.candle
def test_kagi_polars_with_offset_and_fillna() -> None:
    """Test kagi_polars with offset and fillna parameters."""
    df = pl.DataFrame({'close': [100.0, 102.0, 104.0, 106.0]})
    result_df = kagi_polars(
        df, price_col='close', reversal=2.0,
        offset=1, fillna=0.0, output_col='KAGI'
    )
    assert 'KAGI' in result_df.columns
    assert len(result_df) == len(df)
    assert not np.isnan(result_df['KAGI'].to_numpy()).any()


# -----------------------------------------------------------------------------
# Performance tests (benchmarks)
# -----------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.candle
def test_kagi_performance() -> None:
    """Performance benchmark for kagi on large arrays."""
    sizes = [1000, 10000, 100000, 1000000]
    reversal = 2.0
    for size in sizes:
        prices = np.random.randn(size).astype(np.float64) * 10 + 100
        # warm-up
        _ = kagi(prices, reversal)
        # measure
        start = time.perf_counter()
        result = kagi(prices, reversal)
        elapsed = time.perf_counter() - start  # noqa: F841
        # sanity checks
        assert len(result) == size
        assert result.dtype == np.float64
        # Optional: print timing (comment out in CI)
        # print(f"Size {size}: {elapsed:.4f} sec")


# Pure Python reference (slow) - just for illustration
def kagi_python(prices, reversal):  # noqa: D103, C901
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    p0 = prices[0]
    direction = 0
    last_extreme = p0
    for i in range(1, n):
        p = prices[i]
        if direction == 0:
            if p >= p0 + reversal:
                direction = 1
                last_extreme = p
            elif p <= p0 - reversal:
                direction = -1
                last_extreme = p
            out[i] = direction
            continue
        if direction == 1:
            if p > last_extreme:
                last_extreme = p
                out[i] = 1
                continue
            if p <= last_extreme - reversal:
                direction = -1
                last_extreme = p
                out[i] = -1
                continue
            out[i] = 1
        else:
            if p < last_extreme:
                last_extreme = p
                out[i] = -1
                continue
            if p >= last_extreme + reversal:
                direction = 1
                last_extreme = p
                out[i] = 1
                continue
            out[i] = -1
    return out


@pytest.mark.performance
@pytest.mark.candle
def test_kagi_nb_vs_python() -> None:
    """Compare Numba version against a pure-Python reference (if available)."""
    size = 100000
    prices = np.random.randn(size).astype(np.float64) * 10 + 100
    reversal = 2.0
    # Measure Python
    start = time.perf_counter()
    _ = kagi_python(prices, reversal)
    py_time = time.perf_counter() - start
    # Measure Numba (first call includes compilation, so we warm-up)
    _ = kagi(prices[:1000], reversal)
    start = time.perf_counter()
    _ = kagi(prices, reversal)
    nb_time = time.perf_counter() - start
    # Numba should be at least 2x faster on large arrays
    if py_time > 0.01 and nb_time > 0.0:
        assert nb_time < py_time * 0.5, 'Numba version should be faster'
