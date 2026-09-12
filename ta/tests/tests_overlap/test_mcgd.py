# -*- coding: utf-8 -*-
"""Unit tests for MCGD (McGinley Dynamic) module.

Tests cover:
- Numba core function (_mcgd_numba_core)
- Full mcgd_numba with offset, fillna, nan_policy
- Universal wrapper (mcgd_ind) with NumPy and Polars Series input
- Polars integration (mcgd_polars)
- IEEE 754 compliance (NaN, Inf, empty, extreme, zero prices)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.mcgd import (
    _mcgd_numba_core,
    mcgd_numba,
    mcgd_ind,
    mcgd_polars,
)
from ..._array_ops import _apply_offset_fillna


# -----------------------------------------------------------------------------
# Reference implementation (pure Python, mirrors the guarded core)
# -----------------------------------------------------------------------------

def _mcgd_reference(close: np.ndarray, length: int, c: float) -> np.ndarray:
    """Pure-Python reference for the guarded McGinley Dynamic core."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = close[0]
    for i in range(1, n):
        prev = out[i - 1]
        if prev == 0.0:
            out[i] = close[i]
            continue
        ratio = close[i] / prev
        denom = c * length * (ratio ** 4)
        if denom == 0.0:
            out[i] = prev
        else:
            out[i] = prev + (close[i] - prev) / denom
    return out


# -----------------------------------------------------------------------------
# Numba core function tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_mcgd_numba_core_basic() -> None:
    """Test _mcgd_numba_core against the pure-Python reference."""
    close = np.array(
        [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5],
        dtype=np.float64,
    )
    length = 5
    c = 1.0
    result = _mcgd_numba_core(close, length, c)
    expected = _mcgd_reference(close, length, c)
    assert_allclose(result, expected, rtol=1e-12)
    assert result[0] == close[0]


@pytest.mark.overlap
def test_mcgd_numba_core_with_c_multiplier() -> None:
    """Test _mcgd_numba_core with c=0.6 (alternative setting)."""
    close = np.linspace(100.0, 110.0, 20)
    result = _mcgd_numba_core(close, length=10, c=0.6)
    expected = _mcgd_reference(close, 10, 0.6)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_mcgd_numba_basic_random_walk(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test mcgd_numba on the random walk fixture against reference."""
    length = 10
    c = 1.0
    result = mcgd_numba(prices_random_walk, length=length, c=c)
    expected = _mcgd_reference(prices_random_walk, length, c)
    assert result.shape == prices_random_walk.shape
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_mcgd_reacts_to_price_direction() -> None:
    """MCGD follows price direction and remains finite.

    Note: the McGinley formula allows overshoot on sharp gaps by design
    (the adjustment term k*(P/MA - 1) grows with the gap), so we check
    directional reaction and finiteness rather than strict bounds.
    """
    close = np.array([10.0, 12.0, 9.0, 15.0, 8.0, 20.0, 5.0, 25.0])
    result = mcgd_numba(close, length=4)
    assert np.isfinite(result).all()
    # Rising price -> MCGD rises; falling price -> MCGD falls.
    assert result[1] > result[0]
    assert result[2] < result[1]
    # Smooth series: MCGD lags a linear ramp by ~length/2 but stays close.
    smooth = np.linspace(100.0, 110.0, 50)
    smooth_result = mcgd_numba(smooth, length=10)
    assert np.abs(smooth_result[-1] - smooth[-1]) < 3.0


# -----------------------------------------------------------------------------
# Offset, fillna, nan_policy tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_mcgd_numba_offset_fillna() -> None:
    """Test mcgd_numba with offset and fillna using _apply_offset_fillna."""
    close = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 14.0], dtype=np.float64)
    offset = 2
    fillna = 0.0
    base = mcgd_numba(close, length=3, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = mcgd_numba(close, length=3, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.overlap
def test_mcgd_numba_nan_policy_raise() -> None:
    """Test mcgd_numba with nan_policy='raise' raises on NaN."""
    close = np.array([10.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float64)
    with pytest.raises(ValueError, match='contains NaN values'):
        mcgd_numba(close, length=3, nan_policy='raise')


@pytest.mark.overlap
def test_mcgd_numba_nan_policy_ffill() -> None:
    """Test mcgd_numba with nan_policy='ffill' produces finite output."""
    close = np.array([10.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float64)
    result = mcgd_numba(close, length=3, nan_policy='ffill')
    assert np.isfinite(result).all()
    # With ffill, close becomes [10, 11, 11, 13, 14]
    expected = _mcgd_reference(
        np.array([10.0, 11.0, 11.0, 13.0, 14.0]), 3, 1.0
    )
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_mcgd_numba_nan_policy_bfill() -> None:
    """Test mcgd_numba with nan_policy='bfill' produces finite output."""
    close = np.array([10.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float64)
    result = mcgd_numba(close, length=3, nan_policy='bfill')
    assert np.isfinite(result).all()
    # With bfill, close becomes [10, 11, 13, 13, 14]
    expected = _mcgd_reference(
        np.array([10.0, 11.0, 13.0, 13.0, 14.0]), 3, 1.0
    )
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_mcgd_numba_nan_policy_ignore_propagates() -> None:
    """Test mcgd_numba with nan_policy='ignore' propagates NaN onward."""
    close = np.array([10.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float64)
    result = mcgd_numba(close, length=3, nan_policy='ignore')
    assert np.isfinite(result[:2]).all()
    assert np.isnan(result[2:]).all()


@pytest.mark.overlap
def test_mcgd_numba_invalid_nan_policy() -> None:
    """Test mcgd_numba with invalid nan_policy raises ValueError."""
    close = np.array([10.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float64)
    with pytest.raises(ValueError, match='Unknown nan_policy'):
        mcgd_numba(close, length=3, nan_policy='invalid')


@pytest.mark.overlap
def test_mcgd_numba_invalid_length() -> None:
    """Test mcgd_numba with length < 1 raises ValueError."""
    close = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    with pytest.raises(ValueError, match='length must be >= 1'):
        mcgd_numba(close, length=0)


@pytest.mark.overlap
def test_mcgd_numba_invalid_c() -> None:
    """Test mcgd_numba with c <= 0 raises ValueError."""
    close = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    with pytest.raises(ValueError, match='c must be > 0'):
        mcgd_numba(close, length=3, c=0.0)
    with pytest.raises(ValueError, match='c must be > 0'):
        mcgd_numba(close, length=3, c=-1.0)


# -----------------------------------------------------------------------------
# Universal wrapper tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_mcgd_ind_matches_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test mcgd_ind returns the same result as mcgd_numba."""
    result = mcgd_ind(prices_random_walk, length=10)
    expected = mcgd_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_mcgd_ind_with_polars_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test mcgd_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    result = mcgd_ind(s, length=10)
    expected = mcgd_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)


# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_mcgd_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test mcgd_polars returns a DataFrame with a correct MCGD column."""
    length = 10
    result = mcgd_polars(df_random_walk, length=length)
    assert isinstance(result, pl.DataFrame)
    assert f'MCGD_{length}' in result.columns
    close_arr = df_random_walk['close'].to_numpy()
    expected = _mcgd_reference(close_arr, length, 1.0)
    assert_allclose(result[f'MCGD_{length}'].to_numpy(), expected, rtol=1e-12)


@pytest.mark.overlap
def test_mcgd_polars_custom_output_col(df_random_walk: pl.DataFrame) -> None:
    """Test mcgd_polars with a custom output column name."""
    result = mcgd_polars(df_random_walk, length=5, output_col='MCGD')
    assert 'MCGD' in result.columns
    assert result['MCGD'].dtype == pl.Float64


@pytest.mark.overlap
def test_mcgd_polars_with_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test mcgd_polars with offset and fillna."""
    offset = 2
    fillna = 0.0
    close_arr = df_random_walk['close'].to_numpy()
    base = mcgd_numba(close_arr, length=5, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = mcgd_polars(
        df_random_walk,
        length=5,
        offset=offset,
        fillna=fillna,
        output_col='MCGD',
    )
    assert_allclose(result['MCGD'].to_numpy(), expected, rtol=1e-12)


@pytest.mark.overlap
def test_mcgd_polars_custom_close_col(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test mcgd_polars with a non-default close column name."""
    df = pl.DataFrame({'price': prices_random_walk})
    result = mcgd_polars(df, close_col='price', length=10, output_col='MCGD')
    expected = _mcgd_reference(prices_random_walk, 10, 1.0)
    assert_allclose(result['MCGD'].to_numpy(), expected, rtol=1e-12)


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------

def test_mcgd_numba_with_nan(prices_with_nan):
    """NaN in input poisons the recursive filter from that point onward."""
    result = mcgd_numba(prices_with_nan, length=3, nan_policy='ignore')
    assert np.isfinite(result[:5]).all()
    assert np.isnan(result[5:]).all()


def test_mcgd_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so behaves like NaN."""
    result = mcgd_numba(prices_with_inf, length=3, nan_policy='ignore')
    assert np.isfinite(result[:5]).all()
    assert np.isnan(result[5:]).all()


def test_mcgd_numba_empty(prices_empty):
    """Empty input raises ValueError instead of corrupting memory."""
    with pytest.raises(ValueError, match='Input series too short'):
        mcgd_numba(prices_empty, length=3)


def test_mcgd_numba_single(prices_single):
    """Single-element input returns that element (recursion seed)."""
    result = mcgd_numba(prices_single, length=3)
    assert result[0] == prices_single[0]


def test_mcgd_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = mcgd_numba(prices_all_nan, length=3, nan_policy='ignore')
    assert np.isnan(result).all()
    result_fill = mcgd_numba(
        prices_all_nan, length=3, fillna=0.0, nan_policy='ignore'
    )
    assert (result_fill == 0.0).all()


def test_mcgd_numba_extreme_values(prices_extreme):
    """Extreme values must not crash and must stay finite."""
    result = mcgd_numba(prices_extreme, length=3, nan_policy='ignore')
    assert np.isfinite(result).all()
    assert result[0] == prices_extreme[0]


def test_mcgd_numba_zero_price_no_crash():
    """A zero price must not raise ZeroDivisionError (denominator == 0)."""
    close = np.array([10.0, 11.0, 0.0, 12.0, 13.0, 14.0], dtype=np.float64)
    result = mcgd_numba(close, length=3, nan_policy='ignore')
    assert np.isfinite(result).all()
    # Zero-price bar carries the previous value forward.
    assert result[2] == result[1]
    # Filter resumes updating afterwards.
    assert result[3] > result[2]


def test_mcgd_numba_zero_seed():
    """A zero seed re-seeds from the price instead of getting stuck."""
    close = np.array([0.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    result = mcgd_numba(close, length=3, nan_policy='ignore')
    assert np.isfinite(result).all()
    assert result[1] == 10.0


def test_mcgd_polars_with_nan(df_random_walk):
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series('close', close_arr))
    result = mcgd_polars(
        df_with_nan,
        length=3,
        output_col='MCGD',
        nan_policy='ignore',
    )
    vals = result['MCGD'].to_numpy()
    assert np.isfinite(vals[:5]).all()
    assert np.isnan(vals[5:]).all()
