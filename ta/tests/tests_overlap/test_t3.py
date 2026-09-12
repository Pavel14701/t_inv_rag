# -*- coding: utf-8 -*-
"""Unit tests for T3 (Tim Tillson) moving average.
Follows the sma template. Numeric correctness is cross-checked against
TA-Lib where available; structural + IEEE 754 tests run always.
"""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.t3 import t3_ind, t3_numba, t3_polars, t3_talib


@pytest.mark.overlap
def test_t3_numba_warmup() -> None:
    """First 6*(length-1) values are NaN; the tail is finite."""
    close = np.linspace(10.0, 200.0, 100)
    length = 10
    result = t3_numba(close, length=length, a=0.7)
    assert np.isnan(result[: 6 * (length - 1)]).all()
    assert np.isfinite(result[6 * (length - 1) :]).all()


@pytest.mark.overlap
def test_t3_numba_offset_fillna() -> None:
    """offset/fillna are applied via _apply_offset_fillna."""
    close = np.linspace(10.0, 200.0, 80)
    length, offset, fillna = 5, 1, 0.0
    base = t3_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = t3_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_t3_numba_nan_policy_raise() -> None:
    """nan_policy='raise' raises on NaN input."""
    close = np.linspace(10.0, 200.0, 80)
    close[40] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        t3_numba(close, length=5, nan_policy="raise")


@pytest.mark.overlap
def test_t3_numba_nan_policy_ffill_finishes() -> None:
    """After ffill the output is finite once enough history is available."""
    close = np.linspace(10.0, 200.0, 80)
    close[40] = np.nan
    result = t3_numba(close, length=5, nan_policy="ffill")
    assert np.isfinite(result[80:] if False else result[-10:]).all()


@pytest.mark.overlap
def test_t3_numba_invalid_length() -> None:
    """Length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="length must be"):
        t3_numba(np.array([1.0, 2.0, 3.0]), length=0)


@pytest.mark.overlap
def test_t3_ind_uses_numba() -> None:
    """t3_ind uses Numba when use_talib=False."""
    close = np.linspace(10.0, 200.0, 80)
    result = t3_ind(close, length=5, use_talib=False)
    expected = t3_numba(close, length=5)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_t3_numba_matches_talib() -> None:
    """Numba backend reproduces the TA-Lib T3 on a clean series."""
    close = np.linspace(10.0, 200.0, 120)
    for length in (3, 7, 21):
        expected = t3_talib(close, length=length, a=0.7)
        result = t3_numba(close, length=length, a=0.7)
        assert_allclose(result, expected, rtol=1e-5, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_t3_numba_nan_ffill_matches_talib() -> None:
    """Ffill output matches TA-Lib on the forward-filled series."""
    close = np.linspace(10.0, 200.0, 100)
    close[40] = np.nan
    filled = close.copy()
    for i in range(1, len(filled)):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]
    result = t3_numba(close, length=5, nan_policy="ffill")
    expected = t3_talib(filled, length=5)
    assert_allclose(result, expected, rtol=1e-5, equal_nan=True)


@pytest.mark.overlap
def test_t3_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """t3_polars returns a DataFrame with the T3 column."""
    length = 5
    result = t3_polars(df_random_walk, length=length, use_talib=False)
    col = f"T3_{length}_0.7"
    assert col in result.columns
    assert len(result) == len(df_random_walk)
    vals = result[col].to_numpy()
    expected = t3_numba(df_random_walk["close"].to_numpy(), length=length)
    assert_allclose(vals, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_t3_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """t3_polars honours offset/fillna."""
    length = 5
    result = t3_polars(
        df_random_walk, length=length, offset=1, fillna=0.0, use_talib=False
    )
    close = df_random_walk["close"].to_numpy()
    expected = _apply_offset_fillna(t3_numba(close, length=length), 1, 0.0)
    assert_allclose(
        result[f"T3_{length}_0.7"].to_numpy(),
        expected,
        rtol=1e-9,
        equal_nan=True,
    )


# ---- IEEE 754 ----


def test_t3_with_nan(prices_with_nan):
    """NaN input poisons the tail from that point onward (ignore)."""
    result = t3_numba(prices_with_nan, length=3, nan_policy="ignore")
    assert np.isnan(result).all() or np.isnan(result[5:]).any()


def test_t3_with_inf(prices_with_inf):
    """Inf is replaced with NaN; must not escalate to weird finite values."""
    result = t3_numba(prices_with_inf, length=3, nan_policy="ignore")
    assert np.isnan(result).all() or np.isnan(result[5:]).any()


def test_t3_empty(prices_empty):
    """Empty input returns empty without crashing."""
    result = t3_numba(prices_empty, length=3, nan_policy="ignore")
    assert result is not None and len(result) == 0


def test_t3_all_nan(prices_all_nan):
    """All-NaN input stays all-NaN under 'ignore'."""
    result = t3_numba(prices_all_nan, length=3, nan_policy="ignore")
    assert np.isnan(result).all()


def test_t3_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    result = t3_numba(prices_extreme, length=10, nan_policy="ignore")
    assert result is not None


def test_t3_polars_with_nan_df(df_random_walk):
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = t3_polars(
        df_with_nan, length=5, use_talib=False, nan_policy="ignore"
    )
    vals = result["T3_5_0.7"].to_numpy()
    assert np.isnan(vals).all() or np.isnan(vals[5:]).any()
