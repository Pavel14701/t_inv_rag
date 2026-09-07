# -*- coding: utf-8 -*-
"""Unit tests for ATR Trailing Stop (ATRTS) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...volatility.atrs import atrts_numpy, atrts, atrts_polars


def _ohlc_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 17,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Derive realistic high/low/close arrays from the price fixture."""
    rng = np.random.default_rng(seed)
    close = prices_random_walk
    noise = np.abs(rng.normal(0.0, 0.5, len(close))) + 0.1
    high = close + noise
    low = close - noise
    return high, low, close


@pytest.mark.statistics
def test_atrts_numpy_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test atrts_numpy shape and warm-up NaN region."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    length, ma_length = 14, 20
    atrts = atrts_numpy(
        high, low, close, length=length, ma_length=ma_length, use_talib=False,
    )

    assert atrts.shape == close.shape
    assert atrts.dtype == np.float64
    # The stop line starts after both warm-ups: first max(length, ma_length)
    # values are NaN.
    k = max(length, ma_length)
    assert np.isnan(atrts[:k]).all()
    assert np.isfinite(atrts[k:]).all()


@pytest.mark.statistics
def test_atrts_numpy_initial_value(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """The first stop value is close[k] -+ k_mult * atr[k]."""
    from ...volatility.atr import atr_ind
    from ...ma import ma_mode

    high, low, close = _ohlc_arrays(prices_random_walk)
    length, ma_length, k_mult = 14, 20, 3.0
    atrts = atrts_numpy(
        high, low, close, length=length, ma_length=ma_length,
        k=k_mult, mamode='ema', use_talib=False,
    )
    k = max(length, ma_length)
    atr = atr_ind(
        high, low, close, length=length, mamode='ema',
        use_talib=False, nan_policy='ignore',
    )
    ma = ma_mode(
        mamode='ema', source=close, length=ma_length,
        offset=0, fillna=None, use_talib=False,
    )
    if close[k] > ma[k]:
        expected = close[k] - k_mult * atr[k]
    else:
        expected = close[k] + k_mult * atr[k]
    assert_allclose(atrts[k], expected, rtol=1e-9)


@pytest.mark.statistics
def test_atrts_numpy_ratchet_monotone_in_trend() -> None:
    """In a steady uptrend the trailing stop only ratchets upward."""
    n = 80
    close = np.linspace(100.0, 180.0, n)
    noise = np.full(n, 0.5)
    high = close + noise
    low = close - noise
    atrts = atrts_numpy(
        high, low, close, length=5, ma_length=5, k=1.0,
        mamode='sma', use_talib=False,
    )
    stops = atrts[5:]
    assert np.isfinite(stops).all()
    assert (np.diff(stops) >= -1e-9).all()


@pytest.mark.statistics
def test_atrts_numpy_percent_scale(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """percent=True rescales the stop line to 100 * atrts / close."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    base = atrts_numpy(
        high, low, close, length=14, ma_length=20, use_talib=False,
    )
    pct = atrts_numpy(
        high, low, close, length=14, ma_length=20,
        use_talib=False, percent=True,
    )
    k = 20
    expected = base[k:] * 100.0 / close[k:]
    assert_allclose(pct[k:], expected, rtol=1e-12)


@pytest.mark.statistics
def test_atrts_numpy_length_too_short_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 and ma_length < 1 are rejected."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    with pytest.raises(ValueError, match='length must be >= 1'):
        atrts_numpy(high, low, close, length=0, use_talib=False)
    with pytest.raises(ValueError, match='ma_length must be >= 1'):
        atrts_numpy(high, low, close, length=14, ma_length=0, use_talib=False)


@pytest.mark.statistics
def test_atrts_numpy_nan_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN input raises via the ATR backend (nan_policy='raise')."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    close[2] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        atrts_numpy(high, low, close, length=14, ma_length=20, use_talib=False)


@pytest.mark.statistics
def test_atrts_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna on the main line."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    atrts = atrts_numpy(
        high, low, close, length=14, ma_length=20,
        offset=1, fillna=0.0, use_talib=False,
    )
    assert atrts[0] == 0.0


@pytest.mark.statistics
def test_atrts_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test atrts with Polars Series input."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    stop = atrts(
        pl.Series(high), pl.Series(low), pl.Series(close),
        length=14, ma_length=20, use_talib=False,
    )
    assert isinstance(stop, np.ndarray)
    assert np.isfinite(stop[20:]).all()


@pytest.mark.statistics
def test_atrts_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """Test atrts_polars adds the default column."""
    result_df = atrts_polars(df_ohlc, length=14, ma_length=20, use_talib=False)
    assert 'ATRTS_14_20_3.0' in result_df.columns
    assert result_df['ATRTS_14_20_3.0'].dtype == pl.Float64
    assert len(result_df) == len(df_ohlc)

    # Custom output column name.
    custom_df = atrts_polars(
        df_ohlc, length=14, ma_length=20,
        use_talib=False, output_col='STOP',
    )
    assert 'STOP' in custom_df.columns