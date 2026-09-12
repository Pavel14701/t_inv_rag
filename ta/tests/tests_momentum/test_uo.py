# -*- coding: utf-8 -*-
"""Unit tests for Ultimate Oscillator (UO)."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...external import talib, talib_available
from ...momentum.uo import _uo_numba, uo_ind, uo_numpy, uo_polars


@pytest.fixture
def ohlc(prices_random_walk: np.ndarray):
    np.random.seed(42)
    high = prices_random_walk + np.abs(np.random.rand(200)) * 0.5 + 0.1
    low = prices_random_walk - np.abs(np.random.rand(200)) * 0.5 - 0.1
    return high, low, prices_random_walk


def _uo_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fast: int = 7,
    medium: int = 14,
    slow: int = 28,
) -> np.ndarray:
    """Pure numpy UO reference."""
    n = len(close)
    bp = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    bp[1:] = close[1:] - np.minimum(low[1:], close[:-1])
    tr[1:] = (np.maximum(high[1:], close[:-1])
              - np.minimum(low[1:], close[:-1]))
    out = np.full(n, np.nan)
    for i in range(slow, n):
        avgs = []
        for length in (fast, medium, slow):
            s_bp = bp[i - length + 1:i + 1].sum()
            s_tr = tr[i - length + 1:i + 1].sum()
            avgs.append(np.nan if s_tr == 0.0 else s_bp / s_tr)
        out[i] = 100.0 * (4 * avgs[0] + 2 * avgs[1] + avgs[2]) / 7.0
    return out


@pytest.mark.momentum
def test_uo_matches_reference(ohlc) -> None:
    high, low, close = ohlc
    expected = _uo_reference(high, low, close)
    result = uo_numpy(high, low, close, use_talib=False)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_uo_warmup_and_bounds(ohlc) -> None:
    high, low, close = ohlc
    result = uo_numpy(high, low, close, use_talib=False)
    assert np.isnan(result[:28]).all()
    assert not np.isnan(result[28:]).any()
    assert ((result[28:] >= 0.0) & (result[28:] <= 100.0)).all()


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_uo_matches_talib(ohlc) -> None:
    high, low, close = ohlc
    expected = talib.ULTOSC(
        high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28
    )
    result = uo_numpy(high, low, close)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_uo_native_matches_talib_custom_periods(ohlc) -> None:
    high, low, close = ohlc
    expected = talib.ULTOSC(
        high, low, close, timeperiod1=4, timeperiod2=8, timeperiod3=16
    )
    native = uo_numpy(
        high, low, close, fast=4, medium=8, slow=16, use_talib=False
    )
    assert_allclose(native, expected, atol=1e-10, equal_nan=True)


@pytest.mark.momentum
@pytest.mark.momentum
def test_uo_all_rising_close() -> None:
    # Gapping-up bars closing at their high: bp == tr -> UO == 100.
    # (low[i] > close[i-1] and high[i] == close[i] for every i.)
    close = np.arange(1.0, 61.0)
    high = close
    low = close - 0.1
    result = uo_numpy(high, low, close, use_talib=False)
    assert_allclose(result[28:], 100.0, atol=1e-9)


@pytest.mark.momentum
def test_uo_zero_true_range_window_is_nan() -> None:
    # All bars identical: tr == 0 -> averages undefined -> NaN.
    n = 40
    high = np.full(n, 10.0)
    low = np.full(n, 10.0)
    close = np.full(n, 10.0)
    result = uo_numpy(high, low, close, use_talib=False)
    assert np.isnan(result[28:]).all()


@pytest.mark.momentum
def test_uo_nan_propagation(ohlc) -> None:
    high, low, close = ohlc
    close = close.copy()
    close[40] = np.nan
    result = uo_numpy(high, low, close, use_talib=False)
    # bp/tr use close[j] and close[j-1]: both are NaN for j in {40, 41};
    # windows [i-27, i] touching them -> NaN for i in [40, 68].
    assert np.isnan(result[40:69]).all()
    assert not np.isnan(result[28:40]).any()
    assert not np.isnan(result[69:]).any()


@pytest.mark.momentum
@pytest.mark.parametrize('fast, medium, slow', [(0, 14, 28), (7, 0, 28),
                                                (7, 14, 0)])
def test_uo_invalid_params(fast: int, medium: int, slow: int) -> None:
    with pytest.raises(ValueError):
        uo_numpy(
            np.arange(40.0), np.arange(40.0), np.arange(40.0),
            fast=fast, medium=medium, slow=slow,
        )


@pytest.mark.momentum
def test_uo_empty_input() -> None:
    empty = np.array([], dtype=np.float64)
    assert uo_numpy(empty, empty, empty).size == 0


@pytest.mark.momentum
def test_uo_offset_fillna(ohlc) -> None:
    high, low, close = ohlc
    base = uo_numpy(high, low, close, use_talib=False)
    shifted = uo_numpy(
        high, low, close, use_talib=False, offset=2, fillna=50.0
    )
    # fillna replaces both shifted-in positions and warm-up NaNs.
    expected = np.where(np.isnan(base), 50.0, base)
    assert np.all(shifted[:2] == 50.0)
    assert_allclose(shifted[2:], expected[:-2], rtol=1e-12)


@pytest.mark.momentum
def test_uo_ind_numpy_and_series(ohlc) -> None:
    high, low, close = ohlc
    expected = uo_numpy(high, low, close, use_talib=False)
    from_arrays = uo_ind(high, low, close, use_talib=False)
    from_series = uo_ind(
        pl.Series(high), pl.Series(low), pl.Series(close),
        use_talib=False,
    )
    assert_allclose(from_arrays, expected, rtol=1e-12, equal_nan=True)
    assert_allclose(from_series, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_uo_polars(df_ohlc: pl.DataFrame) -> None:
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    expected = uo_numpy(high, low, close, use_talib=False)
    result = uo_polars(df_ohlc, use_talib=False)
    assert 'UO_7_14_28' in result.columns
    assert_allclose(
        result['UO_7_14_28'].to_numpy(), expected,
        rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_uo_readonly_input(ohlc) -> None:
    high, low, close = ohlc
    expected = uo_numpy(high, low, close, use_talib=False)
    for arr in (high, low, close):
        arr.setflags(write=False)
    result = uo_numpy(high, low, close, use_talib=False)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


def test_uo_kernel_bitwise_vs_numpy(ohlc) -> None:
    high, low, close = ohlc
    result = uo_numpy(high, low, close, use_talib=False)
    assert_allclose(
        _uo_numba(high, low, close, 7, 14, 28), result,
        rtol=1e-15, equal_nan=True,
    )
