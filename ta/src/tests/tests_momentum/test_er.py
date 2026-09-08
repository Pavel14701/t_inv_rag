# -*- coding: utf-8 -*-
"""Unit tests for Kaufman Efficiency Ratio (ER)."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ...momentum.er import _er_numba, er_ind, er_numpy, er_polars


def _er_reference(close: np.ndarray, length: int = 10) -> np.ndarray:
    """Pure numpy ER reference."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(length, n):
        change = abs(close[i] - close[i - length])
        vol = np.abs(np.diff(close[i - length:i + 1])).sum()
        if vol != 0.0:
            out[i] = change / vol
    return out


@pytest.mark.momentum
def test_er_matches_reference(prices_random_walk) -> None:
    expected = _er_reference(prices_random_walk, 10)
    result = er_numpy(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_er_warmup_nan() -> None:
    close = np.arange(1.0, 31.0)
    result = er_numpy(close, length=10)
    assert np.isnan(result[:10]).all()
    assert not np.isnan(result[10:]).any()


@pytest.mark.momentum
def test_er_range_and_perfect_trend() -> None:
    # Monotonic series: every diff equal -> ER == 1 exactly.
    close = np.linspace(1.0, 50.0, 30)
    result = er_numpy(close, length=10)
    assert_allclose(result[10:], 1.0, rtol=1e-12)


@pytest.mark.momentum
def test_er_flat_market_is_nan() -> None:
    close = np.full(20, 100.0)
    result = er_numpy(close, length=10)
    # 0/0 -> NaN (undefined), never a fabricated value.
    assert np.isnan(result).all()


@pytest.mark.momentum
def test_er_alternating_series_low_er() -> None:
    # Zigzag: large total volatility, small net change -> ER near 0.
    close = np.tile([100.0, 101.0], 15)
    result = er_numpy(close, length=10)
    assert (result[10:] < 0.2).all()


@pytest.mark.momentum
def test_er_nan_propagation() -> None:
    close = np.arange(1.0, 21.0)
    close[15] = np.nan
    result = er_numpy(close, length=5)
    # Window ending at 15 and all later windows touch the NaN.
    assert np.isnan(result[15:]).all()
    assert not np.isnan(result[5:15]).any()


@pytest.mark.momentum
def test_er_kernel_bitwise_vs_numpy(prices_random_walk) -> None:
    close = prices_random_walk.copy()
    assert_array_equal(_er_numba(close, 7), er_numpy(close, length=7))


@pytest.mark.momentum
def test_er_offset_fillna(prices_random_walk) -> None:
    base = er_numpy(prices_random_walk, length=10)
    shifted = er_numpy(prices_random_walk, length=10, offset=2, fillna=0.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    expected = np.where(np.isnan(base), 0.0, base)
    assert_array_equal(shifted[:2], np.zeros(2))
    assert_array_equal(shifted[2:], expected[:-2])


@pytest.mark.momentum
@pytest.mark.parametrize('length', [0, -1])
def test_er_invalid_length(length: int) -> None:
    with pytest.raises(ValueError, match='length'):
        er_numpy(np.arange(10.0), length=length)


@pytest.mark.momentum
def test_er_empty_input() -> None:
    result = er_numpy(np.array([], dtype=np.float64), length=5)
    assert result.size == 0


@pytest.mark.momentum
def test_er_too_short_input() -> None:
    result = er_numpy(np.array([1.0, 2.0, 3.0]), length=5)
    assert np.isnan(result).all()
    assert result.size == 3


@pytest.mark.momentum
def test_er_ind_numpy_and_series(prices_random_walk) -> None:
    expected = er_numpy(prices_random_walk, length=8)
    from_array = er_ind(prices_random_walk, length=8)
    from_series = er_ind(pl.Series(prices_random_walk), length=8)
    assert_allclose(from_array, expected, rtol=1e-12, equal_nan=True)
    assert_allclose(from_series, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_er_polars(df_random_walk: pl.DataFrame) -> None:
    expected = er_numpy(df_random_walk['close'].to_numpy(), length=10)
    result = er_polars(df_random_walk, length=10)
    assert 'ER_10' in result.columns
    assert_allclose(
        result['ER_10'].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )
    # Original frame is not mutated (new frame returned).
    assert 'ER_10' not in df_random_walk.columns


@pytest.mark.momentum
def test_er_polars_int_column_with_null() -> None:
    df = pl.DataFrame(
        {'close': [1, 2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
        schema={'close': pl.Int64},
    )
    result = er_polars(df, length=5)
    assert result['ER_5'].dtype == pl.Float64
    v = result['ER_5'].to_numpy()
    # Values must match the numpy path on the null -> NaN conversion.
    close_np = df['close'].cast(pl.Float64).to_numpy()
    expected = er_numpy(close_np, length=5)
    assert_allclose(v, expected, rtol=1e-12, equal_nan=True)
    # And the nulls actually propagate as NaN.
    assert np.isnan(v).any()
    assert not np.isinf(v).any()


@pytest.mark.momentum
def test_er_readonly_input(prices_random_walk) -> None:
    arr = prices_random_walk.copy()
    arr.setflags(write=False)
    expected = er_numpy(prices_random_walk, length=6)
    assert_allclose(
        er_numpy(arr, length=6), expected, rtol=1e-12, equal_nan=True
    )
