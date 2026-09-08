# -*- coding: utf-8 -*-
"""Unit tests for SMA (Simple Moving Average) module.

Tests cover:
- Numba core function (_sma_numba_opt)
- Full _sma_numba with offset, fillna, nan_policy, trim
- Backend selection (Numba vs TA-Lib)
- Polars integration (sma_polars)
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.sma import (
    _sma_numba_opt,
    _sma_numba,
    sma_talib,
    sma_ind,
    sma_polars,
)
from ..._array_ops import _apply_offset_fillna
from ...external import talib_available


# -----------------------------------------------------------------------------
# Numba core function tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_sma_numba_opt_basic() -> None:
    """Test _sma_numba_opt on a simple array."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    length = 3
    result = _sma_numba_opt(arr, length)
    expected = np.full_like(arr, np.nan)
    for i in range(length - 1, len(arr)):
        expected[i] = np.mean(arr[i - length + 1: i + 1])
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_numba_basic() -> None:
    """Test _sma_numba (full function) with default parameters."""
    close = np.array(
        [
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0
        ], dtype=np.float64
    )
    length = 3
    result = _sma_numba(close, length=length)
    expected = np.full_like(close, np.nan)
    for i in range(length - 1, len(close)):
        expected[i] = np.mean(close[i - length + 1: i + 1])
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Offset, fillna, nan_policy, trim tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_sma_numba_offset_fillna() -> None:
    """Test _sma_numba with offset and fillna using _apply_offset_fillna."""
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    length = 2
    offset = 1
    fillna = 0.0
    base = _sma_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = _sma_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_numba_nan_policy_raise() -> None:
    """Test _sma_numba with nan_policy='raise' raises on NaN."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match='Input contains NaN values'):
        _sma_numba(close, length=3, nan_policy='raise')


@pytest.mark.overlap
def test_sma_numba_nan_policy_ffill() -> None:
    """Test _sma_numba with nan_policy='ffill'."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    result = _sma_numba(close, length=3, nan_policy='ffill')
    expected = np.full_like(close, np.nan)
    expected[2] = np.mean([1.0, 2.0, 2.0])
    expected[3] = np.mean([2.0, 2.0, 4.0])
    expected[4] = np.mean([2.0, 4.0, 5.0])
    assert_allclose(result[2:], expected[2:], rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_numba_nan_policy_bfill() -> None:
    """Test _sma_numba with nan_policy='bfill'."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    result = _sma_numba(close, length=3, nan_policy='bfill')
    expected = np.full_like(close, np.nan)
    expected[2] = np.mean([1.0, 2.0, 4.0])
    expected[3] = np.mean([2.0, 4.0, 4.0])
    expected[4] = np.mean([4.0, 4.0, 5.0])
    assert_allclose(result[2:], expected[2:], rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_numba_nan_policy_both() -> None:
    """Test _sma_numba with nan_policy='both'."""
    close = np.array([np.nan, 2.0, 3.0, np.nan, 5.0], dtype=np.float64)
    result = _sma_numba(close, length=3, nan_policy='both')
    expected = np.full_like(close, np.nan)
    expected[2] = np.mean([2.0, 2.0, 3.0])
    expected[3] = np.mean([2.0, 3.0, 3.0])
    expected[4] = np.mean([3.0, 3.0, 5.0])
    assert_allclose(result[2:], expected[2:], rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_numba_trim() -> None:
    """Test _sma_numba with trim=True removes first length-1 values."""
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    length = 3
    result = _sma_numba(close, length=length, trim=True)
    expected = np.array([
        np.mean([1.0, 2.0, 3.0]),
        np.mean([2.0, 3.0, 4.0]),
        np.mean([3.0, 4.0, 5.0]),
    ])
    assert len(result) == len(close) - (length - 1)
    assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.overlap
def test_sma_numba_offset_and_trim_error() -> None:
    """Test that offset and trim together raise ValueError."""
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(
        ValueError, match='offset and trim cannot be used simultaneously'
    ):
        _sma_numba(close, length=3, offset=1, trim=True)


# -----------------------------------------------------------------------------
# Backend selection tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_sma_ind_uses_numba() -> None:
    """Test sma_ind uses Numba when use_talib=False."""
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    length = 3
    result = sma_ind(close, length=length, use_talib=False)
    expected = _sma_numba(close, length=length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
@pytest.mark.overlap
def test_sma_ind_uses_talib() -> None:
    """Test sma_ind uses TA-Lib when available and requested."""
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    length = 3
    result = sma_ind(close, length=length, use_talib=True)
    expected = sma_talib(close, length=length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
@pytest.mark.overlap
def test_sma_ind_trim_with_talib_error() -> None:
    """Test that trim=True raises error with TA-Lib backend."""
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match='trim=True is not supported with TA-Lib backend'):
        sma_ind(close, length=3, use_talib=True, trim=True)


# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_sma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test sma_polars returns a Series with correct SMA values."""
    length = 3
    result = sma_polars(df_random_walk, length=length, output_col='SMA', use_talib=False)
    assert result.name == 'SMA'
    assert len(result) == len(df_random_walk)
    assert result.dtype == pl.Float64
    close_arr = df_random_walk['close'].to_numpy()
    expected = np.full_like(close_arr, np.nan)
    for i in range(length - 1, len(close_arr)):
        expected[i] = np.mean(close_arr[i - length + 1: i + 1])
    assert_allclose(result.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test sma_polars with offset and fillna."""
    length = 2
    offset = 1
    fillna = 0.0
    close_arr = df_random_walk['close'].to_numpy()
    base = _sma_numba(close_arr, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = sma_polars(
        df_random_walk,
        length=length,
        offset=offset,
        fillna=fillna,
        output_col='SMA',
        use_talib=False,
    )
    assert_allclose(result.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_polars_nan_policy(df_random_walk: pl.DataFrame) -> None:
    """Test sma_polars with nan_policy='ffill'."""
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[2] = np.nan
    df = df_random_walk.with_columns(pl.Series('close', close_arr))
    length = 3
    result = sma_polars(df, length=length, nan_policy='ffill', output_col='SMA', use_talib=False)
    close_filled = close_arr.copy()
    for i in range(1, len(close_filled)):
        if np.isnan(close_filled[i]):
            close_filled[i] = close_filled[i - 1]
    expected = np.full_like(close_filled, np.nan)
    for i in range(length - 1, len(close_filled)):
        expected[i] = np.mean(close_filled[i - length + 1: i + 1])
    assert_allclose(result.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_polars_uses_numba_by_default(df_random_walk: pl.DataFrame) -> None:
    """Test sma_polars uses Numba by default."""
    length = 3
    result = sma_polars(df_random_walk, length=length, output_col='SMA')
    close_arr = df_random_walk['close'].to_numpy()
    expected = _sma_numba(close_arr, length=length)
    assert_allclose(result.to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_sma_polars_invalid_nan_policy(df_random_walk: pl.DataFrame) -> None:
    """Test sma_polars with invalid nan_policy raises ValueError."""
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[2] = np.nan
    df_nan = df_random_walk.with_columns(pl.Series('close', close_arr))
    with pytest.raises(ValueError, match='Unknown nan_policy'):
        sma_polars(df_nan, length=2, nan_policy='invalid', use_talib=False)


@pytest.mark.overlap
def test_sma_ind_with_polars_series(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test sma_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 3
    result = sma_ind(s, length=length, use_talib=False)
    expected = _sma_numba(prices_random_walk, length=length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------

def test_sma_numba_with_nan(prices_with_nan):
    """NaN only poisons windows containing it; later windows recover.

    NaN is at index 5, length=3 -> windows [3..5], [4..6], [5..7] are
    NaN; from index 8 the window no longer contains it.
    """
    length = 3
    result = _sma_numba(prices_with_nan, length=length, nan_policy='ignore')
    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2:5]).all()
    assert np.isnan(result[5:8]).all()
    assert np.isfinite(result[8:]).all()


def test_sma_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so behaves like NaN."""
    length = 3
    result = _sma_numba(prices_with_inf, length=length, nan_policy='ignore')
    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2:5]).all()
    assert np.isnan(result[5:8]).all()
    assert np.isfinite(result[8:]).all()


def test_sma_numba_empty(prices_empty):
    """Empty input raises ValueError because series too short."""
    with pytest.raises(ValueError, match='Input series too short'):
        _sma_numba(prices_empty, length=3)


def test_sma_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = _sma_numba(prices_all_nan, length=3, nan_policy='ignore')
    assert np.isnan(result).all()
    result_fill = _sma_numba(prices_all_nan, length=3, fillna=0.0, nan_policy='ignore')
    assert (result_fill == 0.0).all()


def test_sma_numba_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    length = 3
    result = _sma_numba(prices_extreme, length=length, nan_policy='ignore')
    assert result is not None


def test_sma_polars_with_nan(df_random_walk):
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series('close', close_arr))
    result = sma_polars(
        df_with_nan,
        length=3,
        output_col='SMA',
        use_talib=False,
        nan_policy='ignore'
    )
    vals = result.to_numpy()
    assert np.isnan(vals[:2]).all()
    assert np.isfinite(vals[2:5]).all()
    # NaN at index 5 poisons only the windows containing it (length=3)
    assert np.isnan(vals[5:8]).all()
    assert np.isfinite(vals[8:]).all()