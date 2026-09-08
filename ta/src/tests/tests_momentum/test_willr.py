# -*- coding: utf-8 -*-
"""Unit tests for Williams %R (WILLR)."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ...external import talib, talib_available
from ...momentum.willr import willr_ind, willr_numpy, willr_polars


@pytest.fixture
def ohlc(prices_random_walk: np.ndarray):
    high = prices_random_walk + np.abs(np.random.rand(200)) * 0.5 + 0.1
    low = prices_random_walk - np.abs(np.random.rand(200)) * 0.5 - 0.1
    return high, low, prices_random_walk


def _willr_reference(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14
) -> np.ndarray:
    """Pure numpy WILLR reference."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        hh = high[i - length + 1:i + 1].max()
        ll = low[i - length + 1:i + 1].min()
        denom = hh - ll
        out[i] = np.nan if denom == 0.0 else -100.0 * (hh - close[i]) / denom
    return out


@pytest.mark.momentum
def test_willr_matches_reference(ohlc) -> None:
    high, low, close = ohlc
    expected = _willr_reference(high, low, close, 14)
    result = willr_numpy(high, low, close, length=14)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_willr_warmup_and_bounds(ohlc) -> None:
    high, low, close = ohlc
    result = willr_numpy(high, low, close, length=14)
    assert np.isnan(result[:13]).all()
    assert not np.isnan(result[13:]).any()
    assert ((result[13:] <= 0.0) & (result[13:] >= -100.0)).all()


@pytest.mark.momentum
def test_willr_close_at_high_is_zero() -> None:
    # Monotone rising series with high == close: the window max of the
    # highs is always the current close -> Williams %R == 0.
    close = np.linspace(1.0, 60.0, 60)
    high = close
    low = close - 1.0
    result = willr_numpy(high, low, close, length=14, use_talib=False)
    assert_allclose(result[13:], 0.0, atol=1e-12)


@pytest.mark.momentum
def test_willr_close_at_low_is_minus_100() -> None:
    # Monotone falling series with low == close: the window min of the
    # lows is always the current close -> Williams %R == -100.
    close = np.linspace(60.0, 1.0, 60)
    high = close + 1.0
    low = close
    result = willr_numpy(high, low, close, length=14, use_talib=False)
    assert_allclose(result[13:], -100.0, atol=1e-12)


@pytest.mark.momentum
def test_willr_flat_window_is_nan() -> None:
    high = np.full(20, 100.0)
    low = np.full(20, 100.0)
    close = np.full(20, 100.0)
    # Native edge rule: flat window (HH == LL) is undefined -> NaN.
    # (TA-Lib returns 0 here, so force the native path.)
    result = willr_numpy(high, low, close, length=14, use_talib=False)
    assert np.isnan(result[13:]).all()


@pytest.mark.momentum
def test_willr_nan_propagation(ohlc) -> None:
    high, low, close = ohlc
    close = close.copy()
    close[20] = np.nan
    result = willr_numpy(high, low, close, length=14, use_talib=False)
    # WILLR reads only the current close: bar 20 itself is NaN while
    # hh/ll come from the (clean) high/low, so all other bars stay
    # finite.
    assert np.isnan(result[20])
    assert not np.isnan(result[14:20]).any()
    assert not np.isnan(result[21:]).any()


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_willr_matches_talib(ohlc) -> None:
    high, low, close = ohlc
    expected = talib.WILLR(high, low, close, timeperiod=14)
    result = willr_numpy(high, low, close, length=14)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_willr_native_matches_talib(ohlc) -> None:
    high, low, close = ohlc
    expected = talib.WILLR(high, low, close, timeperiod=14)
    native = willr_numpy(high, low, close, length=14, use_talib=False)
    assert_allclose(native, expected, atol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_willr_offset_fillna(ohlc) -> None:
    high, low, close = ohlc
    base = willr_numpy(high, low, close, length=14)
    shifted = willr_numpy(high, low, close, length=14, offset=2, fillna=0.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    expected = np.where(np.isnan(base), 0.0, base)
    assert_array_equal(shifted[:2], np.zeros(2))
    assert_array_equal(shifted[2:], expected[:-2])


@pytest.mark.momentum
def test_willr_invalid_length(ohlc) -> None:
    high, low, close = ohlc
    with pytest.raises(ValueError, match='length'):
        willr_numpy(high, low, close, length=0)


@pytest.mark.momentum
def test_willr_empty_input() -> None:
    empty = np.array([], dtype=np.float64)
    assert willr_numpy(empty, empty, empty, length=5).size == 0


@pytest.mark.momentum
def test_willr_ind_numpy_and_series(ohlc) -> None:
    high, low, close = ohlc
    expected = willr_numpy(high, low, close, length=10)
    from_arrays = willr_ind(high, low, close, length=10)
    from_series = willr_ind(
        pl.Series(high), pl.Series(low), pl.Series(close), length=10
    )
    assert_allclose(from_arrays, expected, rtol=1e-12, equal_nan=True)
    assert_allclose(from_series, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_willr_polars(df_ohlc: pl.DataFrame) -> None:
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    expected = willr_numpy(high, low, close, length=14)
    result = willr_polars(df_ohlc, length=14)
    assert 'WILLR_14' in result.columns
    assert_allclose(
        result['WILLR_14'].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )
    assert 'WILLR_14' not in df_ohlc.columns


@pytest.mark.momentum
def test_willr_readonly_input(ohlc) -> None:
    high, low, close = ohlc
    expected = willr_numpy(high, low, close, length=14, use_talib=False)
    for arr in (high, low, close):
        arr.setflags(write=False)
    result = willr_numpy(high, low, close, length=14, use_talib=False)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
