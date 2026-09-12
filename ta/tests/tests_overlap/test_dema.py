# -*- coding: utf-8 -*-
"""Unit tests for Double Exponential Moving Average (DEMA) module.

Tests cover:
- dema_numba against reference implementation
- dema_talib (if TA-Lib available)
- offset and fillna
- dema_ind with Polars Series
- dema_polars DataFrame integration
- Backend selection (use_talib)
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.dema import (
    dema_ind,
    dema_numba,
    dema_polars,
    dema_talib,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------


def _ema_reference(
    close: npt.NDArray[np.float64], length: int
) -> npt.NDArray[np.float64]:
    """Pure Python reference EMA."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    alpha = 2.0 / (length + 1)
    s = 0.0
    for i in range(length):
        s += close[i]
    out[length - 1] = s / length
    for i in range(length, n):
        out[i] = alpha * close[i] + (1 - alpha) * out[i - 1]
    return out


def _dema_reference(
    close: npt.NDArray[np.float64], length: int
) -> npt.NDArray[np.float64]:
    """Pure Python reference DEMA: 2*EMA - EMA(EMA).

    The second EMA is seeded on the valid (non-NaN) part of ema1,
    so DEMA is finite from index 2*(length-1) onward.
    """
    ema1 = _ema_reference(close, length)
    n = len(ema1)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    valid_start = length - 1
    ema2_tail = _ema_reference(ema1[valid_start:], length)
    ema2[2 * valid_start :] = ema2_tail[valid_start:]
    return 2.0 * ema1 - ema2


# -----------------------------------------------------------------------------
# Tests for dema_numba
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_dema_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test dema_numba against pure Python reference."""
    close = prices_random_walk
    length = 10
    result_numba = dema_numba(close, length=length)
    expected = _dema_reference(close, length)
    assert_allclose(result_numba, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_dema_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using the real _apply_offset_fillna."""
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0
    base = dema_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = dema_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_dema_numba_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        dema_numba(close, length=0)


@pytest.mark.overlap
def test_dema_numba_short_window() -> None:
    """Window larger than data returns all NaN (or fillna)."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 5
    result = dema_numba(close, length=length)
    assert np.isnan(result).all()


# -----------------------------------------------------------------------------
# Tests for dema_talib (if available)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_dema_talib_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test dema_talib against pure Python reference."""
    close = prices_random_walk
    length = 10
    result_talib = dema_talib(close, length=length)
    expected = _dema_reference(close, length)
    # TA-Lib uses different initialisation, allow larger tolerance
    mask = np.isfinite(result_talib) & np.isfinite(expected)
    assert_allclose(result_talib[mask], expected[mask], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_dema_talib_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna for TA-Lib version."""
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0
    base = dema_talib(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = dema_talib(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for dema_ind (universal wrapper)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_dema_ind_uses_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test dema_ind uses Numba when TA-Lib is not requested."""
    close = prices_random_walk
    length = 10
    result_numba = dema_ind(close, length=length, use_talib=False)
    expected = _dema_reference(close, length)
    assert_allclose(result_numba, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_dema_ind_uses_talib(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test dema_ind uses TA-Lib when available and requested."""
    close = prices_random_walk
    length = 10
    result_talib = dema_ind(close, length=length, use_talib=True)
    expected = dema_talib(close, length=length)
    assert_allclose(result_talib, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_dema_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test dema_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 10
    result = dema_ind(s, length=length, use_talib=False)
    expected = _dema_reference(prices_random_walk, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for dema_polars (DataFrame integration)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_dema_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test dema_polars adds a column correctly."""
    length = 10
    result_df = dema_polars(
        df_random_walk,
        close_col="close",
        length=length,
        use_talib=False,
        output_col="DEMA",
    )
    assert "DEMA" in result_df.columns
    assert result_df["DEMA"].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk["close"].to_numpy()
    expected = _dema_reference(close_arr, length)
    assert_allclose(
        result_df["DEMA"].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_dema_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    length = 3
    result_df = dema_polars(
        df, close_col="close", length=length, use_talib=False
    )
    expected_col = f"DEMA_{length}"
    assert expected_col in result_df.columns


@pytest.mark.overlap
def test_dema_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test dema_polars with offset and fillna."""
    length = 10
    offset = 3
    fillna = 0.0
    result_df = dema_polars(
        df_random_walk,
        close_col="close",
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=False,
        output_col="DEMA",
    )
    close_arr = df_random_walk["close"].to_numpy()
    expected = dema_numba(
        close_arr, length=length, offset=offset, fillna=fillna
    )
    assert_allclose(
        result_df["DEMA"].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_dema_numba_with_nan(prices_with_nan):
    """NaN in input propagates through EMA and
    remains NaN for all subsequent values.
    """
    length = 5
    result = dema_numba(prices_with_nan, length=length, nan_policy="ignore")
    # NaN at index 5. EMA becomes NaN at index 5 and stays NaN forever.
    # DEMA = 2*EMA1 - EMA2, both become NaN at index 5 and stay NaN.
    # Therefore, result should be NaN from index 5 to the end.
    assert np.isnan(
        result[:5]
    ).all()  # indices 0-4 are NaN due to insufficient data
    assert np.isnan(result[5:]).all()  # from 5 onward all NaN


@pytest.mark.overlap
def test_dema_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    length = 5
    result = dema_numba(prices_with_inf, length=length, nan_policy="ignore")
    # Same as NaN test
    assert np.isnan(result[:5]).all()
    assert np.isnan(result[5:]).all()


@pytest.mark.overlap
def test_dema_numba_empty(prices_empty):
    """Empty input returns empty array."""
    result = dema_numba(prices_empty, length=5)
    assert result.size == 0


@pytest.mark.overlap
def test_dema_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = dema_numba(prices_all_nan, length=5, nan_policy="ignore")
    assert np.isnan(result).all()
    result_fill = dema_numba(
        prices_all_nan, length=5, fillna=0.0, nan_policy="ignore"
    )
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_dema_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    length = 5
    result = dema_numba(prices_extreme, length=length, nan_policy="ignore")
    assert result is not None


@pytest.mark.overlap
def test_dema_numba_nan_policy_raise() -> None:
    """Input with NaN and default nan_policy='raise' raises ValueError."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="NaN"):
        dema_numba(data, length=3)


@pytest.mark.overlap
def test_dema_numba_nan_policy_ffill() -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    result = dema_numba(data, length=3, nan_policy="ffill")
    # after warmup all values must be finite
    assert np.isfinite(result[4:]).all()


@pytest.mark.overlap
def test_dema_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly."""
    # Create a copy of the close column with NaN inserted at index 5
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([pl.Series("close", close_arr)])
    result_df = dema_polars(
        df_with_nan,
        close_col="close",
        length=5,
        output_col="DEMA",
        nan_policy="ignore",
    )
    assert "DEMA" in result_df.columns
    assert len(result_df) == len(df_random_walk)
    dema_vals = result_df["DEMA"].to_numpy()
    # After NaN appears, DEMA should remain NaN forever
    assert np.isnan(dema_vals[5:]).all()
