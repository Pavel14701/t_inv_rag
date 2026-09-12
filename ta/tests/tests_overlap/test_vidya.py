# -*- coding: utf-8 -*-
"""Unit tests for VIDYA (Variable Index Dynamic Average) module.

Tests cover:
- Numba core function (_cmo_numba)
- Full vidya_numba with offset, fillna, nan_policy
- Backend selection (Numba vs TA-Lib)
- Polars integration (vidya_polars)
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.vidya import (
    _cmo_numba,
    vidya_ind,
    vidya_numba,
    vidya_polars,
    vidya_talib,
)


def _vidya_reference(close, length=10, drift=1):
    """Pure-NumPy reference of VIDYA matching the implementation exactly."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length + drift:
        return out
    alpha = 2.0 / (length + 1.0)
    diff = np.zeros(n, dtype=np.float64)
    for i in range(drift, n):
        diff[i] = close[i] - close[i - drift]
    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)
    cum_pos = np.zeros(n + 1, dtype=np.float64)
    cum_neg = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        cum_pos[i] = cum_pos[i - 1] + pos[i - 1]
        cum_neg[i] = cum_neg[i - 1] + neg[i - 1]
    cmo = np.full(n, np.nan, dtype=np.float64)
    for i in range(length + drift - 1, n):
        pos_sum = cum_pos[i + 1] - cum_pos[i - length + 1]
        neg_sum = cum_neg[i + 1] - cum_neg[i - length + 1]
        denom = pos_sum + neg_sum
        cmo[i] = (pos_sum - neg_sum) / denom if denom != 0.0 else 0.0
    start = length + drift - 1
    vidya = np.full(n, np.nan, dtype=np.float64)
    if not np.isnan(cmo[start]) and not np.isnan(close[start]):
        vidya[start] = close[start]
        for i in range(start + 1, n):
            if np.isnan(cmo[i]) or np.isnan(close[i]):
                break
            sc = alpha * abs(cmo[i])
            vidya[i] = sc * close[i] + (1.0 - sc) * vidya[i - 1]
    out[:start] = np.nan
    out[start:] = vidya[start:]
    return out


@pytest.mark.overlap
def test_cmo_numba_basic() -> None:
    """Test _cmo_numba against a rolling reference on a simple array."""
    arr = np.linspace(1.0, 50.0, 50)
    length, drift = 10, 1
    result = _cmo_numba(arr, length, drift)
    diff = np.full(len(arr), np.nan)
    diff[drift:] = arr[drift:] - arr[:-drift]
    up = np.where(np.isnan(diff), 0.0, np.maximum(diff, 0.0))
    dn = np.where(np.isnan(diff), 0.0, np.maximum(-diff, 0.0))
    c_up = np.concatenate(([0.0], np.cumsum(up)))
    c_dn = np.concatenate(([0.0], np.cumsum(dn)))
    expected = np.full(len(arr), np.nan)
    for i in range(length + drift - 1, len(arr)):
        pos = c_up[i] - c_up[i - length]
        neg = c_dn[i] - c_dn[i - length]
        denom = pos + neg
        expected[i] = (pos - neg) / denom if denom != 0.0 else 0.0
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_vidya_numba_basic() -> None:
    """Test vidya_numba (full function) with default parameters."""
    close = np.linspace(100.0, 200.0, 50)
    length = 10
    result = vidya_numba(close, length=length)
    expected = _vidya_reference(close, length=length)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    assert np.isnan(result[:length]).all()
    assert np.isfinite(result[length:]).all()


@pytest.mark.overlap
def test_vidya_numba_uptrend_lag() -> None:
    """In a steady uptrend VIDYA must stay below price and track it."""
    close = np.linspace(100.0, 200.0, 60)
    result = vidya_numba(close, length=10)
    body = result[10:]
    # seed: vidya[10] == close[10]; from then on it lags strictly below
    assert body[0] == close[10]
    assert (body[1:] < close[11:]).all()
    assert body[-1] > close[-1] - 10.0


@pytest.mark.overlap
def test_vidya_numba_offset_fillna() -> None:
    """Test vidya_numba with offset and fillna using _apply_offset_fillna."""
    close = np.linspace(100.0, 150.0, 30)
    length, offset, fillna = 10, 1, 0.0
    base = vidya_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = vidya_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_vidya_numba_negative_offset() -> None:
    """Negative offset shifts the result backward."""
    close = np.linspace(100.0, 150.0, 30)
    base = vidya_numba(close, length=10, offset=0, fillna=None)
    result = vidya_numba(close, length=10, offset=-2, fillna=None)
    expected = _apply_offset_fillna(base, -2, None)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_vidya_numba_nan_policy_raise() -> None:
    """Test vidya_numba with nan_policy='raise' raises on NaN."""
    close = np.linspace(100.0, 150.0, 30)
    close[15] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        vidya_numba(close, length=10, nan_policy="raise")


@pytest.mark.overlap
def test_vidya_numba_nan_policy_ffill() -> None:
    """Ffill fills the NaN with the previous close before the recursion."""
    close = np.linspace(100.0, 150.0, 30)
    close[15] = np.nan
    filled = close.copy()
    for i in range(1, len(filled)):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]
    result = vidya_numba(close, length=10, nan_policy="ffill")
    expected = _vidya_reference(filled, length=10)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_vidya_numba_nan_policy_bfill() -> None:
    """Bfill fills the NaN with the next close before the recursion."""
    close = np.linspace(100.0, 150.0, 30)
    close[15] = np.nan
    filled = close.copy()
    for i in range(len(filled) - 2, -1, -1):
        if np.isnan(filled[i]):
            filled[i] = filled[i + 1]
    result = vidya_numba(close, length=10, nan_policy="bfill")
    expected = _vidya_reference(filled, length=10)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_vidya_numba_nan_policy_both() -> None:
    """'both' fills leading NaNs backward and inner NaNs forward."""
    close = np.linspace(100.0, 150.0, 30)
    close[0] = np.nan
    close[15] = np.nan
    filled = close.copy()
    for i in range(1, len(filled)):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]
    for i in range(len(filled) - 2, -1, -1):
        if np.isnan(filled[i]):
            filled[i] = filled[i + 1]
    result = vidya_numba(close, length=10, nan_policy="both")
    expected = _vidya_reference(filled, length=10)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_vidya_numba_invalid_nan_policy() -> None:
    """Invalid nan_policy raises ValueError."""
    close = np.linspace(100.0, 150.0, 30)
    close[15] = np.nan
    with pytest.raises(ValueError, match="Unknown nan_policy"):
        vidya_numba(close, length=10, nan_policy="invalid")


@pytest.mark.overlap
def test_vidya_numba_invalid_length() -> None:
    """Length < 1 raises ValueError."""
    close = np.linspace(100.0, 150.0, 30)
    with pytest.raises(ValueError, match="length must be"):
        vidya_numba(close, length=0)


@pytest.mark.overlap
def test_vidya_numba_too_short_series() -> None:
    """Series shorter than length + drift raises ValueError."""
    close = np.linspace(100.0, 110.0, 5)
    with pytest.raises(ValueError, match="Input series too short"):
        vidya_numba(close, length=10)


@pytest.mark.overlap
def test_vidya_ind_uses_numba() -> None:
    """Test vidya_ind uses Numba when use_talib=False."""
    close = np.linspace(100.0, 200.0, 50)
    result = vidya_ind(close, length=10, use_talib=False)
    expected = vidya_numba(close, length=10)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_vidya_ind_uses_talib() -> None:
    """Test vidya_ind uses TA-Lib CMO when available and requested."""
    close = np.linspace(100.0, 200.0, 60)
    result = vidya_ind(close, length=10, use_talib=True)
    expected = vidya_talib(close, length=10)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    assert np.isfinite(result[15:]).all()
    numba_res = vidya_numba(close, length=10)
    assert np.isfinite(numba_res[15:]).all()


@pytest.mark.overlap
def test_vidya_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test vidya_polars returns a DataFrame with correct column."""
    length = 10
    result = vidya_polars(df_random_walk, length=length, use_talib=False)
    assert "VIDYA_10" in result.columns
    assert len(result) == len(df_random_walk)
    assert result["VIDYA_10"].dtype == pl.Float64
    vals = result["VIDYA_10"].to_numpy()
    assert np.isnan(vals[:length]).all()
    assert np.isfinite(vals[length:]).all()


@pytest.mark.overlap
def test_vidya_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test vidya_polars with offset and fillna."""
    length = 10
    offset = 1
    fillna = 0.0
    result = vidya_polars(
        df_random_walk,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=False,
    )
    vals = result["VIDYA_10"].to_numpy()
    expected = _apply_offset_fillna(
        vidya_numba(df_random_walk["close"].to_numpy(), length=length),
        offset,
        fillna,
    )
    assert_allclose(vals, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_vidya_polars_custom_output_col(df_random_walk: pl.DataFrame) -> None:
    """Test vidya_polars with custom output column name."""
    length = 14
    result = vidya_polars(
        df_random_walk,
        length=length,
        output_col="CUSTOM_VIDYA",
        use_talib=False,
    )
    assert "CUSTOM_VIDYA" in result.columns
    assert result["CUSTOM_VIDYA"].dtype == pl.Float64


@pytest.mark.overlap
def test_vidya_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """Test vidya_polars propagates NaN correctly."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    close_arr[15] = np.inf
    df_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = vidya_polars(
        df_nan, length=10, use_talib=False, nan_policy="ignore"
    )
    vals = result["VIDYA_10"].to_numpy()
    assert np.isnan(vals).all()


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests
# -----------------------------------------------------------------------------


def test_vidya_with_nan(prices_with_nan):
    """NaN in input makes VIDYA NaN from that point onward."""
    length = 5
    result = vidya_numba(prices_with_nan, length=length, nan_policy="ignore")
    assert np.isnan(result[:length]).all()
    assert np.isnan(result[5:]).all()


def test_vidya_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so behaves like NaN."""
    length = 5
    result = vidya_numba(prices_with_inf, length=length, nan_policy="ignore")
    assert np.isnan(result[:length]).all()
    assert np.isnan(result[5:]).all()


def test_vidya_empty(prices_empty):
    """Empty input raises ValueError because series too short."""
    length = 5
    with pytest.raises(ValueError, match="Input series too short"):
        vidya_numba(prices_empty, length=length)


def test_vidya_all_nan(prices_all_nan):
    """All NaNs raise under default 'raise'; 'ignore' propagates them."""
    length = 5
    with pytest.raises(ValueError, match="contains NaN"):
        vidya_numba(prices_all_nan, length=length)
    result = vidya_numba(prices_all_nan, length=length, nan_policy="ignore")
    assert np.isnan(result).all()


def test_vidya_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    length = 10
    result = vidya_numba(prices_extreme, length=length, nan_policy="ignore")
    assert result is not None


def test_vidya_polars_with_nan_df(df_random_walk):
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    close_arr[15] = np.inf
    df_with_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = vidya_polars(
        df_with_nan, length=10, use_talib=False, nan_policy="ignore"
    )
    vals = result["VIDYA_10"].to_numpy()
    assert np.isnan(vals).all()
