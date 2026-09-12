# -*- coding: utf-8 -*-
"""Unit tests for Pretty Good Oscillator (PGO)."""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from ta.src.momentum.pgo import pgo_ind, pgo_numpy, pgo_polars
from ta.src.overlap.ema import ema_ind


@pytest.fixture
def ohlc(prices_random_walk: np.ndarray):
    np.random.seed(42)
    high = prices_random_walk + np.abs(np.random.rand(200)) * 0.5 + 0.1
    low = prices_random_walk - np.abs(np.random.rand(200)) * 0.5 - 0.1
    return high, low, prices_random_walk


def _pgo_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
) -> np.ndarray:
    """Pure numpy PGO reference."""
    n = len(close)
    ema = ema_ind(close, length=length, use_talib=False, nan_policy="ignore")
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        denom = (
            high[i - length + 1 : i + 1].max()
            - low[i - length + 1 : i + 1].min()
        )
        out[i] = np.nan if denom == 0.0 else (close[i] - ema[i]) / denom
    return out


@pytest.mark.momentum
def test_pgo_matches_reference(ohlc) -> None:
    high, low, close = ohlc
    expected = _pgo_reference(high, low, close, 14)
    result = pgo_numpy(high, low, close, length=14)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_pgo_warmup_nan(ohlc) -> None:
    high, low, close = ohlc
    result = pgo_numpy(high, low, close, length=14)
    assert np.isnan(result[:13]).all()
    assert not np.isnan(result[13:]).any()


@pytest.mark.momentum
def test_pgo_zero_when_close_equals_ema(ohlc) -> None:
    # Flat market: close == EMA(close) -> numerator zero, denom > 0.
    high = np.linspace(1.0, 2.0, 30)  # strictly growing range
    low = high - 1.0
    close = np.full(30, 1.5)
    result = pgo_numpy(high, low, close, length=5)
    assert_allclose(result[4:], 0.0, atol=1e-12)


@pytest.mark.momentum
def test_pgo_flat_window_is_nan() -> None:
    high = np.full(20, 100.0)
    low = np.full(20, 100.0)
    close = np.full(20, 100.0)
    result = pgo_numpy(high, low, close, length=14)
    # HH == LL -> 0/0 -> NaN (native edge rule; EMA never sees it).
    assert np.isnan(result[13:]).all()


@pytest.mark.momentum
def test_pgo_breakout_magnitude(ohlc) -> None:
    high, low, close = ohlc
    # Strong final rally: PGO must turn clearly positive.
    close = close.copy()
    close[-5:] = close[-6] + np.array([3.0, 6.0, 9.0, 12.0, 15.0])
    high = close + 0.5
    low = close - 0.5
    result = pgo_numpy(high, low, close, length=14)
    assert result[-1] > 0.0
    assert result[-1] > result[-6]


@pytest.mark.momentum
def test_pgo_ema_component_matches_ema_ind(ohlc) -> None:
    high, low, close = ohlc
    ema = ema_ind(close, length=14, use_talib=False, nan_policy="ignore")
    high_max = np.array(
        [high[max(0, i - 13) : i + 1].max() for i in range(len(high))]
    )
    low_min = np.array(
        [low[max(0, i - 13) : i + 1].min() for i in range(len(low))]
    )
    expected = np.where(
        high_max - low_min == 0.0,
        np.nan,
        (close - ema) / (high_max - low_min),
    )
    result = pgo_numpy(high, low, close, length=14, use_talib=False)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_pgo_invalid_length(ohlc) -> None:
    high, low, close = ohlc
    with pytest.raises(ValueError, match="length"):
        pgo_numpy(high, low, close, length=0)


@pytest.mark.momentum
def test_pgo_empty_input() -> None:
    empty = np.array([], dtype=np.float64)
    assert pgo_numpy(empty, empty, empty, length=5).size == 0


@pytest.mark.momentum
def test_pgo_offset_fillna(ohlc) -> None:
    high, low, close = ohlc
    base = pgo_numpy(high, low, close, length=14)
    shifted = pgo_numpy(high, low, close, length=14, offset=2, fillna=0.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    expected = np.where(np.isnan(base), 0.0, base)
    assert_array_equal(shifted[:2], np.zeros(2))
    assert_array_equal(shifted[2:], expected[:-2])


@pytest.mark.momentum
def test_pgo_ind_numpy_and_series(ohlc) -> None:
    high, low, close = ohlc
    expected = pgo_numpy(high, low, close, length=10)
    from_arrays = pgo_ind(high, low, close, length=10)
    from_series = pgo_ind(
        pl.Series(high), pl.Series(low), pl.Series(close), length=10
    )
    assert_allclose(from_arrays, expected, rtol=1e-12, equal_nan=True)
    assert_allclose(from_series, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_pgo_polars(df_ohlc: pl.DataFrame) -> None:
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()
    expected = pgo_numpy(high, low, close, length=14)
    result = pgo_polars(df_ohlc, length=14)
    assert "PGO_14" in result.columns
    assert_allclose(
        result["PGO_14"].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


@pytest.mark.momentum
def test_pgo_readonly_input(ohlc) -> None:
    high, low, close = ohlc
    expected = pgo_numpy(high, low, close, length=14, use_talib=False)
    high.setflags(write=False)
    low.setflags(write=False)
    result = pgo_numpy(high, low, close, length=14, use_talib=False)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
