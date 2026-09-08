# -*- coding: utf-8 -*-
"""Unit tests for Fisher Transform."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ...momentum.fisher import (
    _fisher_numba,
    fisher_ind,
    fisher_numpy,
    fisher_polars,
)


@pytest.fixture
def ohlc(prices_random_walk: np.ndarray):
    np.random.seed(42)
    high = prices_random_walk + np.abs(np.random.rand(200)) * 0.5 + 0.1
    low = prices_random_walk - np.abs(np.random.rand(200)) * 0.5 - 0.1
    close = prices_random_walk
    return high, low, close


def _fisher_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 9,
) -> np.ndarray:
    """Slow pure-python reference (window max/min + recursion)."""
    n = len(close)
    fisher = np.full(n, np.nan)
    raw_prev = 0.0
    value_prev = 0.0
    started = False
    for i in range(length - 1, n):
        hh = high[i - length + 1:i + 1].max()
        ll = low[i - length + 1:i + 1].min()
        denom = hh - ll
        if denom == 0.0 or np.isnan(denom):
            if started:
                raw_prev = np.nan
                value_prev = np.nan
            continue
        hlc3 = (high[i] + low[i] + close[i]) / 3.0
        raw = 0.66 * ((hlc3 - ll) / denom - 0.5) + 0.67 * raw_prev
        value = 0.5 * (raw + value_prev)
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
        raw_prev = raw
        value_prev = value
        started = True
    return fisher


@pytest.mark.momentum
def test_fisher_matches_reference(ohlc) -> None:
    high, low, close = ohlc
    expected = _fisher_reference(high, low, close, 9)
    fisher, signal = fisher_numpy(high, low, close, length=9)
    assert_allclose(fisher, expected, rtol=1e-12, equal_nan=True)
    # Signal is Fisher shifted by one bar.
    assert np.isnan(signal[0])
    assert_allclose(signal[1:], expected[:-1], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_fisher_warmup_nan(ohlc) -> None:
    high, low, close = ohlc
    fisher, signal = fisher_numpy(high, low, close, length=9)
    assert np.isnan(fisher[:8]).all()
    assert not np.isnan(fisher[8:]).any()
    assert np.isnan(signal[:9]).all()


@pytest.mark.momentum
def test_fisher_flat_window_propagates_nan() -> None:
    n = 30
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    close = np.full(n, 100.0)
    # Create one moving bar first so the recursion starts, then flatten.
    close[0:10] = np.linspace(99.0, 100.0, 10)
    high[0:10] = close[0:10] + 0.5
    low[0:10] = close[0:10] - 0.5
    fisher, _ = fisher_numpy(high, low, close, length=5)
    assert np.isnan(fisher[14:]).all()  # flat window poisons the recursion


@pytest.mark.momentum
def test_fisher_large_move_sign(ohlc) -> None:
    high, low, close = ohlc
    # Strong final up-move: Fisher should be clearly positive at the end.
    close = close.copy()
    close[-5:] = close[-6] + np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    high = close + 0.5
    low = close - 0.5
    fisher, _ = fisher_numpy(high, low, close, length=9)
    assert fisher[-1] > 0.5


@pytest.mark.momentum
def test_fisher_kernel_vs_numpy(ohlc) -> None:
    high, low, close = ohlc
    hlc3 = (high + low + close) / 3.0
    hh = np.array([high[max(0, i - 8):i + 1].max() for i in range(len(high))])
    ll = np.array([low[max(0, i - 8):i + 1].min() for i in range(len(low))])
    fisher, _ = fisher_numpy(high, low, close, length=9)
    assert_array_equal(_fisher_numba(hlc3, hh, ll, 9), fisher)


@pytest.mark.momentum
def test_fisher_invalid_length(ohlc) -> None:
    high, low, close = ohlc
    with pytest.raises(ValueError, match='length'):
        fisher_numpy(high, low, close, length=0)


@pytest.mark.momentum
def test_fisher_empty_input() -> None:
    empty = np.array([], dtype=np.float64)
    fisher, signal = fisher_numpy(empty, empty, empty, length=5)
    assert fisher.size == 0
    assert signal.size == 0


@pytest.mark.momentum
def test_fisher_offset_fillna(ohlc) -> None:
    high, low, close = ohlc
    base, _ = fisher_numpy(high, low, close, length=9)
    shifted, _ = fisher_numpy(high, low, close, length=9,
                              offset=3, fillna=0.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    expected = np.where(np.isnan(base), 0.0, base)
    assert_array_equal(shifted[:3], np.zeros(3))
    assert_array_equal(shifted[3:], expected[:-3])


@pytest.mark.momentum
def test_fisher_ind_numpy_and_series(ohlc) -> None:
    high, low, close = ohlc
    expected = fisher_numpy(high, low, close, length=9)
    from_arrays = fisher_ind(high, low, close, length=9)
    from_series = fisher_ind(
        pl.Series(high), pl.Series(low), pl.Series(close), length=9
    )
    for res, exp in zip(from_arrays, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    for res, exp in zip(from_series, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_fisher_polars(df_ohlc: pl.DataFrame) -> None:
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    fisher, signal = fisher_numpy(high, low, close, length=9)
    result = fisher_polars(df_ohlc, length=9)
    assert 'FISHERT_9' in result.columns
    assert 'FISHERTs_9' in result.columns
    assert_allclose(
        result['FISHERT_9'].to_numpy(), fisher, rtol=1e-12, equal_nan=True
    )
    assert_allclose(
        result['FISHERTs_9'].to_numpy(), signal, rtol=1e-12, equal_nan=True
    )


@pytest.mark.momentum
def test_fisher_readonly_input(ohlc) -> None:
    high, low, close = ohlc
    expected = fisher_numpy(high, low, close, length=9)
    for arr in (high, low):
        arr.setflags(write=False)
    fisher, signal = fisher_numpy(high, low, close, length=9)
    assert_allclose(fisher, expected[0], rtol=1e-12, equal_nan=True)
    assert_allclose(signal, expected[1], rtol=1e-12, equal_nan=True)
