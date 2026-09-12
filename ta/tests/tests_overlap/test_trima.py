# -*- coding: utf-8 -*-
"""Unit tests for TRIMA (Triangular Moving Average).
test sma template (offset, fillna, nan_policy, backend, polars, IEEE).
"""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.trima import (
    trima_ind,
    trima_numba,
    trima_polars,
    trima_talib,
)


def _sma_np(arr: np.ndarray, w: int) -> np.ndarray:
    """Simple moving average matching the Numba core.

    A window containing NaN yields NaN; later windows recover
    (mirrors ``_sma_numba_opt``).
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    for i in range(w - 1, n):
        window = arr[i + 1 - w : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = window.sum() / w
    return out


def _trima_reference(close: np.ndarray, length: int) -> np.ndarray:
    """Double-SMA triangular moving average matching trima_numba exactly."""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    trima = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return trima
    if length % 2 == 1:
        h = (length + 1) // 2
        sma1 = _sma_np(close, h)
        sma2_tail = _sma_np(sma1[h - 1 :], h)
        trima[length - 1 :] = sma2_tail[h - 1 :]
    else:
        h = length // 2
        sma1 = _sma_np(close, h)
        sma2_tail = _sma_np(sma1[h - 1 :], h + 1)
        trima[length - 1 :] = sma2_tail[h:]
    return trima


@pytest.mark.overlap
def test_trima_numba_basic() -> None:
    """trima_numba matches the double-SMA reference (odd and even)."""
    close = np.linspace(10.0, 120.0, 60)
    for length in (3, 5, 10, 21):
        result = trima_numba(close, length=length)
        expected = _trima_reference(close, length)
        assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
        # warmup is the first length-1 values
        assert np.isnan(result[: length - 1]).all()


@pytest.mark.overlap
def test_trima_numba_offset_fillna() -> None:
    """offset/fillna go through _apply_offset_fillna."""
    close = np.linspace(10.0, 120.0, 40)
    length, offset, fillna = 5, 1, 0.0
    base = trima_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = trima_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_trima_numba_nan_policy_raise() -> None:
    """nan_policy='raise' raises on NaN input."""
    close = np.linspace(10.0, 120.0, 40)
    close[15] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        trima_numba(close, length=5, nan_policy="raise")


@pytest.mark.overlap
def test_trima_numba_nan_policy_ffill() -> None:
    """Ffill matches the reference on the forward-filled series."""
    close = np.linspace(10.0, 120.0, 40)
    close[15] = np.nan
    filled = close.copy()
    for i in range(1, len(filled)):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]
    result = trima_numba(close, length=5, nan_policy="ffill")
    expected = _trima_reference(filled, 5)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_trima_numba_invalid_length() -> None:
    """Length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="length must be"):
        trima_numba(np.array([1.0, 2.0, 3.0]), length=0)


@pytest.mark.overlap
def test_trima_ind_uses_numba() -> None:
    """trima_ind uses Numba when use_talib=False."""
    close = np.linspace(10.0, 120.0, 50)
    result = trima_ind(close, length=5, use_talib=False)
    expected = trima_numba(close, length=5)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_trima_ind_uses_talib() -> None:
    """trima_ind with TA-Lib matches trima_talib on a clean series."""
    close = np.linspace(10.0, 120.0, 60)
    result = trima_ind(close, length=7, use_talib=True)
    expected = trima_talib(close, length=7)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_trima_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """trima_polars returns a DataFrame with the TRIMA column."""
    length = 5
    result = trima_polars(df_random_walk, length=length, use_talib=False)
    col = f"TRIMA_{length}"
    assert col in result.columns
    assert len(result) == len(df_random_walk)
    vals = result[col].to_numpy()
    expected = _trima_reference(df_random_walk["close"].to_numpy(), length)
    assert_allclose(vals, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_trima_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """trima_polars honours offset/fillna."""
    length = 5
    result = trima_polars(
        df_random_walk, length=length, offset=1, fillna=0.0, use_talib=False
    )
    close = df_random_walk["close"].to_numpy()
    expected = _apply_offset_fillna(_trima_reference(close, length), 1, 0.0)
    assert_allclose(
        result["TRIMA_5"].to_numpy(), expected, rtol=1e-9, equal_nan=True
    )


# ---- IEEE 754 ----


def test_trima_with_nan(prices_with_nan):
    """NaN only poisons TRIMA windows containing it (ignore)."""
    result = trima_numba(prices_with_nan, length=3, nan_policy="ignore")
    expected = _trima_reference(prices_with_nan, 3)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    assert np.isnan(result[:2]).all()
    assert np.isnan(result[5:]).any()
    assert np.isfinite(result[8:]).all()


def test_trima_with_inf(prices_with_inf):
    """Inf is replaced with NaN, so behaves like NaN."""
    result = trima_numba(prices_with_inf, length=3, nan_policy="ignore")
    cleaned = prices_with_inf.copy()
    cleaned[~np.isfinite(cleaned)] = np.nan
    expected = _trima_reference(cleaned, 3)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    assert np.isfinite(result[8:]).all()


def test_trima_empty(prices_empty):
    """Empty input returns an empty array (or NaN pattern) without crash."""
    result = trima_numba(prices_empty, length=3, nan_policy="ignore")
    assert result is not None


def test_trima_all_nan(prices_all_nan):
    """All-NaN input stays all-NaN under 'ignore'."""
    result = trima_numba(prices_all_nan, length=3, nan_policy="ignore")
    assert np.isnan(result).all()


def test_trima_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    result = trima_numba(prices_extreme, length=10, nan_policy="ignore")
    assert result is not None
