# -*- coding: utf-8 -*-
"""Unit tests for LINREG (Linear Regression) indicator.
Follows the sma template: numeric reference, offset/fillna, nan_policy,
backend selection, polars integration, IEEE 754 compliance.
"""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.linreg import (
    _linreg_numba_core,
    linreg_ind,
    linreg_numba,
    linreg_polars,
    linreg_talib,
)


MODE_LIST = ("line", "tsf", "slope", "intercept", "angle", "r")


def _linreg_reference(close, length, mode="line", degrees=False):
    """Linear regression from first principles, matching the Numba core."""
    close = np.asarray(close, dtype=np.float64).copy()
    close[~np.isfinite(close)] = np.nan
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    x = np.arange(1, length + 1, dtype=np.float64)
    x_sum = x.sum()
    x2_sum = (x * x).sum()
    divisor = length * x2_sum - x_sum * x_sum
    inv = 1.0 / divisor if divisor != 0.0 else 0.0
    for i in range(length - 1, n):
        y = close[i - length + 1 : i + 1]
        y_sum = y.sum()
        xy_sum = np.dot(x, y)
        slope = (length * xy_sum - x_sum * y_sum) * inv
        if mode == "slope":
            out[i] = slope
            continue
        intercept = (y_sum - slope * x_sum) / length
        if mode == "intercept":
            out[i] = intercept
            continue
        if mode == "angle":
            ang = np.arctan(slope)
            out[i] = ang * (180.0 / np.pi) if degrees else ang
            continue
        if mode == "r":
            y2_sum = (y * y).sum()
            denom = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
            out[i] = (
                (length * xy_sum - x_sum * y_sum) / denom
                if denom != 0.0
                else 0.0
            )
            continue
        x_last = length + 1.0 if mode == "tsf" else length
        out[i] = slope * x_last + intercept
    return out


@pytest.mark.overlap
def test_linreg_core_matches_reference() -> None:
    """Numba core matches the first-principles reference for every mode."""
    close = np.linspace(10.0, 200.0, 100) + np.sin(np.linspace(0, 6, 100))
    for mode in MODE_LIST:
        for degrees in (False, True):
            result = _linreg_numba_core(
                close.astype(np.float64), 10, mode, degrees
            )
            expected = _linreg_reference(close, 10, mode, degrees)
            assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_linreg_numba_matches_reference() -> None:
    """linreg_numba matches the reference for every mode."""
    close = np.linspace(10.0, 200.0, 100)
    for mode in MODE_LIST:
        result = linreg_numba(close, length=10, mode=mode)  # type: ignore[arg-type]
        expected = _linreg_reference(close, 10, mode, False)
        assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_linreg_numba_angle_degrees() -> None:
    """Angle mode honours the degrees flag."""
    close = np.linspace(10.0, 200.0, 100)
    rad = linreg_numba(close, length=10, mode="angle")
    deg = linreg_numba(close, length=10, mode="angle", degrees=True)
    assert_allclose(deg, rad * (180.0 / np.pi), rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_linreg_numba_offset_fillna() -> None:
    """offset/fillna are applied via _apply_offset_fillna."""
    close = np.linspace(10.0, 200.0, 60)
    length, offset, fillna = 5, 1, 0.0
    base = linreg_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = linreg_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_linreg_numba_nan_policy_raise() -> None:
    """nan_policy='raise' raises on NaN input."""
    close = np.linspace(10.0, 200.0, 60)
    close[30] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        linreg_numba(close, length=5, nan_policy="raise")


@pytest.mark.overlap
def test_linreg_numba_invalid_mode() -> None:
    """Unknown mode raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported mode"):
        linreg_numba(
            np.linspace(1.0, 50.0, 50),
            length=5,
            mode="bogus",  # type: ignore[arg-type]
        )


@pytest.mark.overlap
def test_linreg_numba_invalid_length() -> None:
    """Length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="length must be"):
        linreg_numba(np.linspace(1.0, 50.0, 50), length=0)


@pytest.mark.overlap
def test_linreg_ind_uses_numba() -> None:
    """linreg_ind uses Numba for 'r' mode regardless of use_talib."""
    close = np.linspace(10.0, 200.0, 60)
    result = linreg_ind(close, length=5, mode="r", use_talib=True)
    expected = linreg_numba(close, length=5, mode="r")
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_linreg_ind_uses_talib_line() -> None:
    """linreg_ind with TA-Lib 'line' matches linreg_talib."""
    close = np.linspace(10.0, 200.0, 80)
    result = linreg_ind(close, length=7, mode="line", use_talib=True)
    expected = linreg_talib(close, 7, "line")
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_linreg_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """linreg_polars returns a DataFrame with the LINREG column."""
    length, mode = 5, "line"
    result = linreg_polars(
        df_random_walk,
        length=length,
        mode=mode,  # type: ignore[arg-type]
        use_talib=False,
    )
    col = f"LINREG_{mode}_{length}"
    assert col in result.columns
    assert len(result) == len(df_random_walk)
    expected = _linreg_reference(
        df_random_walk["close"].to_numpy(), length, mode
    )
    assert_allclose(
        result[col].to_numpy(), expected, rtol=1e-9, equal_nan=True
    )


@pytest.mark.overlap
def test_linreg_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """linreg_polars honours offset/fillna."""
    length = 5
    result = linreg_polars(
        df_random_walk, length=length, offset=1, fillna=0.0, use_talib=False
    )
    expected = _apply_offset_fillna(
        _linreg_reference(df_random_walk["close"].to_numpy(), length), 1, 0.0
    )
    assert_allclose(
        result["LINREG_line_5"].to_numpy(), expected, rtol=1e-9, equal_nan=True
    )


# ---- IEEE 754 ----


def test_linreg_with_nan(prices_with_nan):
    """NaN input poisons only windows containing it; then recovers."""
    length = 3
    result = linreg_numba(prices_with_nan, length=length, nan_policy="ignore")
    expected = _linreg_reference(prices_with_nan, length)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    # warmup NaN
    assert np.isnan(result[: length - 1]).all()
    # windows containing index 5 are NaN (indices 5..5+length-1)
    assert np.isnan(result[5 : 5 + length]).all()
    # fully valid windows after the NaN recover
    assert np.isfinite(result[5 + length :]).all()


def test_linreg_with_inf(prices_with_inf):
    """Inf is replaced with NaN (matches the sanitised reference)."""
    length = 3
    result = linreg_numba(prices_with_inf, length=length, nan_policy="ignore")
    cleaned = prices_with_inf.copy()
    cleaned[~np.isfinite(cleaned)] = np.nan
    expected = _linreg_reference(cleaned, length)
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)
    assert np.isnan(result[5 : 5 + length]).all()
    assert np.isfinite(result[5 + length :]).all()


def test_linreg_empty(prices_empty):
    """Empty input returns empty without crashing."""
    result = linreg_numba(prices_empty, length=3, nan_policy="ignore")
    assert result is not None and len(result) == 0


def test_linreg_all_nan(prices_all_nan):
    """All-NaN input stays all-NaN under 'ignore'."""
    result = linreg_numba(prices_all_nan, length=3, nan_policy="ignore")
    assert np.isnan(result).all()


def test_linreg_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    result = linreg_numba(prices_extreme, length=10, nan_policy="ignore")
    assert result is not None


def test_linreg_polars_with_nan_df(df_random_walk):
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = linreg_polars(
        df_with_nan, length=5, use_talib=False, nan_policy="ignore"
    )
    vals = result["LINREG_line_5"].to_numpy()
    assert np.isnan(vals[5:]).any()
