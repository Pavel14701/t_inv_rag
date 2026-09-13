# -*- coding: utf-8 -*-
"""Unit tests for full Stochastic Oscillator (STOCH)."""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _rolling_max_numba, _rolling_min_numba
from ta.src.external import talib, talib_available
from ta.src.ma import ma_mode
from ta.src.momentum.stoch import stoch_ind, stoch_numpy, stoch_polars


@pytest.fixture
def ohlc(prices_random_walk: np.ndarray):
    np.random.seed(42)
    high = prices_random_walk + np.abs(np.random.rand(200)) * 0.5 + 0.1
    low = prices_random_walk - np.abs(np.random.rand(200)) * 0.5 - 0.1
    return high, low, prices_random_walk


def _stoch_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure numpy STOCH reference (SMA smoothing)."""
    n = len(close)
    raw = np.full(n, np.nan)
    for i in range(k - 1, n):
        hh = high[i - k + 1 : i + 1].max()
        ll = low[i - k + 1 : i + 1].min()
        denom = hh - ll
        raw[i] = np.nan if denom == 0.0 else 100.0 * (close[i] - ll) / denom
    k_line = np.full(n, np.nan)
    for i in range(k + smooth_k - 2, n):
        window = raw[i - smooth_k + 1 : i + 1]
        if not np.isnan(window).any():
            k_line[i] = window.mean()
    d_line = np.full(n, np.nan)
    for i in range(k + smooth_k + d - 3, n):
        window = k_line[i - d + 1 : i + 1]
        if not np.isnan(window).any():
            d_line[i] = window.mean()
    return k_line, d_line


@pytest.mark.momentum
def test_stoch_matches_reference(ohlc) -> None:
    high, low, close = ohlc
    expected_k, expected_d = _stoch_reference(high, low, close)
    stoch_k, stoch_d = stoch_numpy(high, low, close, use_talib=False)
    assert_allclose(stoch_k, expected_k, rtol=1e-12, equal_nan=True)
    assert_allclose(stoch_d, expected_d, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stoch_warmup_and_bounds(ohlc) -> None:
    high, low, close = ohlc
    stoch_k, stoch_d = stoch_numpy(high, low, close, use_talib=False)
    # raw valid from 13; %K (sma 3) from 15; %D (sma 3) from 17.
    assert np.isnan(stoch_k[:15]).all()
    assert not np.isnan(stoch_k[15:]).any()
    assert np.isnan(stoch_d[:17]).all()
    assert ((stoch_k[15:] >= 0.0) & (stoch_k[15:] <= 100.0)).all()
    assert ((stoch_d[17:] >= 0.0) & (stoch_d[17:] <= 100.0)).all()


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
def test_stoch_matches_talib(ohlc) -> None:
    high, low, close = ohlc
    expected_k, expected_d = talib.STOCH(
        high,
        low,
        close,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=talib.MA_Type.SMA,
        slowd_period=3,
        slowd_matype=talib.MA_Type.SMA,
    )
    stoch_k, stoch_d = stoch_numpy(high, low, close)
    assert_allclose(stoch_k, expected_k, rtol=1e-10, equal_nan=True)
    assert_allclose(stoch_d, expected_d, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
def test_stoch_native_matches_talib(ohlc) -> None:
    high, low, close = ohlc
    talib_k, talib_d = talib.STOCH(
        high,
        low,
        close,
        fastk_period=10,
        slowk_period=5,
        slowk_matype=talib.MA_Type.SMA,
        slowd_period=4,
        slowd_matype=talib.MA_Type.SMA,
    )
    native_k, native_d = stoch_numpy(
        high, low, close, k=10, d=4, smooth_k=5, use_talib=False
    )
    # TA-Lib prepends extra warm-up NaNs; compare the common region.
    mask = ~np.isnan(talib_k)
    assert_allclose(native_k[mask], talib_k[mask], rtol=1e-10)
    mask_d = ~np.isnan(talib_d)
    assert_allclose(native_d[mask_d], talib_d[mask_d], rtol=1e-10)


@pytest.mark.momentum
def test_stoch_ema_mode_differs_from_sma(ohlc) -> None:
    high, low, close = ohlc
    k_sma, _ = stoch_numpy(high, low, close, use_talib=False, mamode="sma")
    k_ema, _ = stoch_numpy(high, low, close, use_talib=False, mamode="ema")
    assert not np.allclose(k_sma[20:], k_ema[20:], rtol=1e-6)
    # EMA mode still matches ma_mode-based reference.
    ll = _rolling_min_numba(np.ascontiguousarray(low.copy()), 14)
    hh = _rolling_max_numba(np.ascontiguousarray(high.copy()), 14)
    denom = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100.0 * (close - ll) / denom
    raw = np.where(denom == 0.0, np.nan, raw)
    expected = ma_mode(
        "ema", raw, length=3, use_talib=False, nan_policy="ignore"
    )
    assert_allclose(k_ema, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stoch_flat_window_is_nan() -> None:
    n = 30
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    close = np.full(n, 100.0)
    stoch_k, stoch_d = stoch_numpy(high, low, close, use_talib=False)
    # raw is 0/0 (undefined) on every window -> everything NaN.
    assert np.isnan(stoch_k).all()
    assert np.isnan(stoch_d).all()


@pytest.mark.momentum
def test_stoch_nan_propagation(ohlc) -> None:
    high, low, close = ohlc
    close = close.copy()
    close[30] = np.nan
    stoch_k, _stoch_d = stoch_numpy(high, low, close, use_talib=False)
    # raw[i] windows [i-13, i] touch bar 30 only at i = 30; the
    # nan-ignore smoothing keeps exactly the %K windows [30, 32] NaN.
    assert np.isnan(stoch_k[30:33]).all()
    assert not np.isnan(stoch_k[15:30]).any()
    assert not np.isnan(stoch_k[33:]).any()


@pytest.mark.momentum
@pytest.mark.parametrize(
    "kw",
    [
        {"k": 0},
        {"d": -1},
        {"smooth_k": 0},
    ],
)
def test_stoch_invalid_params(kw: dict) -> None:
    with pytest.raises(ValueError):
        stoch_numpy(np.arange(40.0), np.arange(40.0), np.arange(40.0), **kw)


@pytest.mark.momentum
def test_stoch_empty_input() -> None:
    empty = np.array([], dtype=np.float64)
    stoch_k, stoch_d = stoch_numpy(empty, empty, empty, use_talib=False)
    assert stoch_k.size == 0
    assert stoch_d.size == 0


@pytest.mark.momentum
def test_stoch_offset_fillna(ohlc) -> None:
    high, low, close = ohlc
    base_k, base_d = stoch_numpy(high, low, close, use_talib=False)
    k, d = stoch_numpy(
        high, low, close, use_talib=False, offset=2, fillna=50.0
    )
    # fillna replaces both shifted-in positions and warm-up NaNs.
    exp_k = np.where(np.isnan(base_k), 50.0, base_k)
    exp_d = np.where(np.isnan(base_d), 50.0, base_d)
    assert np.all(k[:2] == 50.0)
    assert_allclose(k[2:], exp_k[:-2], rtol=1e-12)
    assert_allclose(d[2:], exp_d[:-2], rtol=1e-12)


@pytest.mark.momentum
def test_stoch_ind_numpy_and_series(ohlc) -> None:
    high, low, close = ohlc
    expected = stoch_numpy(high, low, close, use_talib=False)
    from_arrays = stoch_ind(high, low, close, use_talib=False)
    from_series = stoch_ind(
        pl.Series(high),
        pl.Series(low),
        pl.Series(close),
        use_talib=False,
    )
    for res, exp in zip(from_arrays, expected, strict=False):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    for res, exp in zip(from_series, expected, strict=False):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stoch_polars(df_ohlc: pl.DataFrame) -> None:
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()
    stoch_k, stoch_d = stoch_numpy(high, low, close, use_talib=False)
    result = stoch_polars(df_ohlc, use_talib=False)
    assert "STOCHk_14_3_3" in result.columns
    assert "STOCHd_14_3_3" in result.columns
    assert_allclose(
        result["STOCHk_14_3_3"].to_numpy(),
        stoch_k,
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        result["STOCHd_14_3_3"].to_numpy(),
        stoch_d,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_stoch_readonly_input(ohlc) -> None:
    high, low, close = ohlc
    expected = stoch_numpy(high, low, close, use_talib=False)
    high = high.copy()
    low = low.copy()
    high.setflags(write=False)
    low.setflags(write=False)
    stoch_k, stoch_d = stoch_numpy(high, low, close, use_talib=False)
    assert_allclose(stoch_k, expected[0], rtol=1e-12, equal_nan=True)
    assert_allclose(stoch_d, expected[1], rtol=1e-12, equal_nan=True)
