# -*- coding: utf-8 -*-
"""Unit tests for Stochastic RSI module.

Covers:
- formula parity stoch(RSI) with MA smoothing
- warm-up NaNs end at rsi_length + length + k - 2
- flat RSI window (denominator zero) -> neutral 50.0, no warnings
- regression: smoothing no longer raises on the inherent warm-up NaNs,
  and runs on the numpy backend (TA-Lib SMA never recovers after NaN)
- parameter validation
- trim semantics
- offset / fillna semantics
- stochrsi_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.ma import ma_mode
from ta.src.momentum.rsi import rsi_ind
from ta.src.momentum.stochrsi import (
    stochrsi_ind,
    stochrsi_numpy,
    stochrsi_polars,
)


def _np(arr: object) -> np.ndarray:
    assert isinstance(arr, np.ndarray)
    return arr


def _stoch_reference(
    close: npt.NDArray[np.float64],
    rsi_length: int,
    length: int,
) -> np.ndarray:
    """Raw stochastic of RSI: 100 * (rsi - min) / (max - min)."""
    rsi = _np(rsi_ind(close, length=rsi_length, use_talib=False))
    out = np.full(len(close), np.nan)
    for i in range(rsi_length + length - 2, len(close)):
        w = rsi[i - length + 1 : i + 1]
        lo, hi = w.min(), w.max()
        out[i] = 50.0 if hi == lo else 100.0 * (rsi[i] - lo) / (hi - lo)
    return out


@pytest.mark.momentum
def test_stochrsi_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy backend equals stoch(RSI) smoothed with SMA(k)/SMA(d)."""
    close = prices_random_walk
    length, rsi_length, k, d = 14, 14, 3, 3
    stoch_k, stoch_d = stochrsi_numpy(
        close,
        length=length,
        rsi_length=rsi_length,
        k=k,
        d=d,
        use_talib=False,
    )
    stoch = _stoch_reference(close, rsi_length, length)
    expected_k = _np(ma_mode("sma", stoch, length=k, nan_policy="ignore"))
    expected_d = _np(ma_mode("sma", expected_k, length=d, nan_policy="ignore"))
    assert_allclose(
        stoch_k, expected_k, rtol=1e-10, atol=1e-10, equal_nan=True
    )
    assert_allclose(
        stoch_d, expected_d, rtol=1e-10, atol=1e-10, equal_nan=True
    )


@pytest.mark.momentum
def test_stochrsi_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """%K is NaN until index 27, finite after.

    rsi_ind warms up at index rsi_length - 2 = 12, the rolling
    min/max window adds length - 1 = 13 and the %K SMA adds k - 1 = 2.
    """
    rsi_length, length, k = 14, 14, 3
    stoch_k, _ = stochrsi_numpy(
        prices_random_walk,
        length=length,
        rsi_length=rsi_length,
        k=k,
        use_talib=False,
    )
    warm = 28
    assert np.isnan(stoch_k[:warm]).all()
    assert np.isfinite(stoch_k[warm:]).all()


@pytest.mark.momentum
def test_stochrsi_no_raise_on_warmup(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Regression: smoothing must not raise on inherent warm-up NaNs.

    Previously ma_mode received nan_policy='raise' and stochrsi_numpy
    raised ValueError for every input.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stoch_k, stoch_d = stochrsi_numpy(prices_random_walk, use_talib=False)
    assert np.isfinite(stoch_k[28:]).all()
    assert np.isfinite(stoch_d[30:]).all()


@pytest.mark.momentum
def test_stochrsi_flat_rsi_neutral(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """A flat RSI window (denominator zero) yields the neutral 50.0.

    Strictly constant prices give RSI = NaN (0/0), so we use a constant
    uptick instead: RSI settles at a flat 100.0 and every stochastic
    window has denominator zero.
    """
    n = len(prices_random_walk)
    close = 100.0 + 0.1 * np.arange(n)
    stoch_k, stoch_d = stochrsi_numpy(close, use_talib=False)
    assert np.allclose(stoch_k[28:], 50.0)
    assert np.allclose(stoch_d[30:], 50.0)


@pytest.mark.momentum
def test_stochrsi_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Periods < 1 are rejected."""
    for kwargs in (dict(length=0), dict(rsi_length=0), dict(k=0), dict(d=0)):
        with pytest.raises(ValueError, match="must be >= 1"):
            stochrsi_numpy(prices_random_walk, **kwargs)


@pytest.mark.momentum
def test_stochrsi_trim(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """trim=True shortens both outputs identically, starting at 1st valid."""
    k_full, d_full = stochrsi_numpy(prices_random_walk, use_talib=False)
    k_trim, d_trim = stochrsi_numpy(
        prices_random_walk, trim=True, use_talib=False
    )
    first = int(np.argmax(np.isfinite(d_full)))
    assert len(k_trim) == len(k_full) - first
    assert len(d_trim) == len(d_full) - first
    assert_allclose(d_trim, d_full[first:], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stochrsi_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    k0, d0 = stochrsi_numpy(prices_random_walk, use_talib=False)
    k2, d2 = stochrsi_numpy(
        prices_random_walk,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert k2[0] == 0.0 and k2[1] == 0.0
    assert d2[0] == 0.0
    # %K finite from 28, %D from 30 -> shifted comparison starts at 30/32
    assert_allclose(k2[30:], k0[28:-2], rtol=1e-12, equal_nan=True)
    assert_allclose(d2[32:], d0[30:-2], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stochrsi_ind_accepts_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """stochrsi_ind accepts numpy arrays and polars Series alike."""
    expected = stochrsi_numpy(prices_random_walk, use_talib=False)
    result = stochrsi_ind(pl.Series(prices_random_walk), use_talib=False)
    assert_allclose(result[0], expected[0], rtol=1e-12, equal_nan=True)
    assert_allclose(result[1], expected[1], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stochrsi_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """stochrsi_polars adds STOCHRSIk/STOCHRSId columns."""
    result = stochrsi_polars(df_ohlc, use_talib=False)
    assert "STOCHRSIk_14_14_3_3" in result.columns
    assert "STOCHRSId_14_14_3_3" in result.columns
    assert len(result) == len(df_ohlc)
    expected_k, expected_d = stochrsi_numpy(
        df_ohlc["close"].to_numpy(),
        use_talib=False,
    )
    assert_allclose(
        result["STOCHRSIk_14_14_3_3"].to_numpy(), expected_k, equal_nan=True
    )
    assert_allclose(
        result["STOCHRSId_14_14_3_3"].to_numpy(), expected_d, equal_nan=True
    )
