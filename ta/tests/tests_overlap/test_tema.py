# -*- coding: utf-8 -*-
"""Unit tests for TEMA (Triple Exponential Moving Average).
Follows the sma template (reference, offset/fillna, nan_policy, backend,
polars, IEEE 754).
"""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.ema import _ema_numba_opt
from ta.src.overlap.tema import tema_ind, tema_numba, tema_polars, tema_talib


# ---- IEEE 754 helpers ----
def _fin(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite (Inf/NaN) with NaN."""
    out = arr.copy()
    out[~np.isfinite(out)] = np.nan
    return out


def _tema_reference(close: np.ndarray, length: int) -> np.ndarray:
    """TEMA = 3*(EMA - EMA2) + EMA3 with valid-seed alignment."""
    close = np.asarray(close, dtype=np.float64).copy()
    close[~np.isfinite(close)] = np.nan
    n = len(close)
    vs = length - 1
    ema1 = _ema_numba_opt(close, length)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    ema3 = np.full(n, np.nan, dtype=np.float64)
    if n > vs:
        ema2_tail = _ema_numba_opt(np.ascontiguousarray(ema1[vs:]), length)
        ema2[2 * vs :] = ema2_tail[vs:]
        if n > 2 * vs:
            ema3_tail = _ema_numba_opt(
                np.ascontiguousarray(ema2[2 * vs :]), length
            )
            ema3[3 * vs :] = ema3_tail[vs:]
    return 3.0 * (ema1 - ema2) + ema3


@pytest.mark.overlap
def test_tema_numba_basic() -> None:
    """tema_numba matches the EMA-composition reference."""
    close = np.linspace(10.0, 200.0, 80)
    for length in (3, 5, 10):
        result = tema_numba(close, length=length)
        expected = _tema_reference(close, length)
        assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
        assert np.isnan(result[: 3 * (length - 1)]).all()


@pytest.mark.overlap
def test_tema_numba_offset_fillna() -> None:
    """offset/fillna are applied via _apply_offset_fillna."""
    close = np.linspace(10.0, 200.0, 60)
    length, offset, fillna = 5, 1, 0.0
    base = tema_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = tema_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_tema_numba_nan_policy_raise() -> None:
    """nan_policy='raise' raises on NaN input."""
    close = np.linspace(10.0, 200.0, 60)
    close[20] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        tema_numba(close, length=5, nan_policy="raise")


@pytest.mark.overlap
def test_tema_numba_nan_policy_ffill() -> None:
    """Ffill matches the reference on the forward-filled series."""
    close = np.linspace(10.0, 200.0, 60)
    close[20] = np.nan
    filled = close.copy()
    for i in range(1, len(filled)):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]
    result = tema_numba(close, length=5, nan_policy="ffill")
    expected = _tema_reference(filled, 5)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_tema_numba_invalid_length() -> None:
    """Length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="length must be"):
        tema_numba(np.array([1.0, 2.0, 3.0]), length=0)


@pytest.mark.overlap
def test_tema_ind_uses_numba() -> None:
    """tema_ind uses Numba when use_talib=False."""
    close = np.linspace(10.0, 200.0, 60)
    result = tema_ind(close, length=5, use_talib=False)
    expected = tema_numba(close, length=5)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_tema_ind_uses_talib() -> None:
    """tema_ind with TA-Lib matches tema_talib on a clean series."""
    close = np.linspace(10.0, 200.0, 80)
    result = tema_ind(close, length=7, use_talib=True)
    expected = tema_talib(close, length=7)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_tema_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """tema_polars returns a DataFrame with the TEMA column."""
    length = 5
    result = tema_polars(df_random_walk, length=length, use_talib=False)
    col = f"TEMA_{length}"
    assert col in result.columns
    assert len(result) == len(df_random_walk)
    vals = result[col].to_numpy()
    expected = _tema_reference(df_random_walk["close"].to_numpy(), length)
    assert_allclose(vals, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_tema_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """tema_polars honours offset/fillna."""
    length = 5
    result = tema_polars(
        df_random_walk, length=length, offset=1, fillna=0.0, use_talib=False
    )
    close = df_random_walk["close"].to_numpy()
    expected = _apply_offset_fillna(_tema_reference(close, length), 1, 0.0)
    assert_allclose(
        result["TEMA_5"].to_numpy(), expected, rtol=1e-9, equal_nan=True
    )


# ---- IEEE 754 ----


def test_tema_with_nan(prices_with_nan):
    """NaN input poisons TEMA from that point onward (ignore)."""
    result = tema_numba(prices_with_nan, length=3, nan_policy="ignore")
    expected = _tema_reference(prices_with_nan, 3)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    assert np.isnan(result[5:]).any()


def test_tema_with_inf(prices_with_inf):
    """Inf is replaced with NaN (matches reference on sanitised input)."""
    result = tema_numba(prices_with_inf, length=3, nan_policy="ignore")
    expected = _tema_reference(prices_with_inf, 3)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    assert np.isnan(result[5:]).any()


def test_tema_empty(prices_empty):
    """Empty input returns empty without crashing."""
    result = tema_numba(prices_empty, length=3, nan_policy="ignore")
    assert result is not None


def test_tema_all_nan(prices_all_nan):
    """All-NaN input stays all-NaN under 'ignore'."""
    result = tema_numba(prices_all_nan, length=3, nan_policy="ignore")
    assert np.isnan(result).all()


def test_tema_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    result = tema_numba(prices_extreme, length=10, nan_policy="ignore")
    assert result is not None


def test_tema_polars_with_nan_df(df_random_walk):
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = tema_polars(
        df_with_nan, length=5, use_talib=False, nan_policy="ignore"
    )
    vals = result["TEMA_5"].to_numpy()
    expected = _tema_reference(close_arr, 5)
    assert_allclose(vals, expected, rtol=1e-9, equal_nan=True)
    assert np.isnan(vals[5:]).any()
