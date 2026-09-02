# -*- coding: utf-8 -*-
"""Unit tests for Smoothed Moving Average (SMMA) module.

Tests cover:
- Numba core function (_smma_numba_core)
- smma_numba with offset/fillna
- smma_ind with Polars Series
- smma_polars DataFrame integration
- Comparison with reference implementation
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.smma import (
    _smma_numba_core,
    smma_numba,
    smma_ind,
    smma_polars,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------

def _smma_reference(
    close: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure Python reference implementation of SMMA."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial SMA
    s = 0.0
    for i in range(length):
        s += close[i]
    out[length - 1] = s / length
    # Recurrence
    for i in range(length, n):
        out[i] = ((length - 1) * out[i - 1] + close[i]) / length
    return out


# -----------------------------------------------------------------------------
# Tests for _smma_numba_core
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_smma_numba_core_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test _smma_numba_core against the pure Python reference."""
    close = prices_random_walk
    length = 10
    result = _smma_numba_core(close, length)
    expected = _smma_reference(close, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_smma_numba_core_short_window() -> None:
    """Window longer than data returns all NaN."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 5
    result = _smma_numba_core(close, length)
    assert np.isnan(result).all()


@pytest.mark.overlap
def test_smma_numba_core_single_value() -> None:
    """Length 1 returns the series itself."""
    close = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    length = 1
    result = _smma_numba_core(close, length)
    assert_allclose(result, close, rtol=1e-6)


# -----------------------------------------------------------------------------
# Helper to apply offset+fillna consistently with the real implementation
# -----------------------------------------------------------------------------

def _apply_offset_fillna_test(
    arr: np.ndarray,
    offset: int,
    fillna: float,
) -> np.ndarray:
    """Apply shift and fillna exactly as in the real implementation."""
    out = np.full_like(arr, np.nan)
    if offset > 0:
        out[offset:] = arr[:-offset]
    elif offset < 0:
        out[:offset] = arr[-offset:]
    else:
        out[:] = arr
    out[np.isnan(out)] = fillna
    return out


# -----------------------------------------------------------------------------
# Tests for smma_numba with offset/fillna
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_smma_numba_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test smma_numba against the pure Python reference."""
    close = prices_random_walk
    length = 10
    result = smma_numba(close, length=length)
    expected = _smma_reference(close, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_smma_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna by comparing with manually applied shift+fillna."""  # noqa: E501
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0

    base = smma_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna_test(base, offset, fillna)

    result = smma_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_smma_numba_negative_offset(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test negative offset (backward shift)."""
    close = prices_random_walk
    length = 10
    offset = -3
    fillna = -1.0

    base = smma_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna_test(base, offset, fillna)

    result = smma_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for smma_ind (universal wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_smma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test smma_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 10
    result = smma_ind(s, length=length)
    expected = _smma_reference(prices_random_walk, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for smma_polars (DataFrame integration)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_smma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test smma_polars adds a column correctly."""
    length = 10
    result_df = smma_polars(
        df_random_walk, close_col='close', length=length, output_col='SMMA'
    )
    assert 'SMMA' in result_df.columns
    assert result_df['SMMA'].dtype == pl.Float64
    close_arr = df_random_walk['close'].to_numpy()
    expected = _smma_reference(close_arr, length)
    assert_allclose(
        result_df['SMMA'].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_smma_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame(
        {'close': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    length = 3
    result_df = smma_polars(df, close_col='close', length=length)
    assert f'SMMA_{length}' in result_df.columns


@pytest.mark.overlap
def test_smma_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """Test smma_polars with offset and fillna."""
    length = 10
    offset = 3
    fillna = 0.0
    result_df = smma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        offset=offset,
        fillna=fillna,
        output_col='SMMA',
    )
    close_arr = df_random_walk['close'].to_numpy()
    expected = smma_numba(
        close_arr, length=length, offset=offset, fillna=fillna
    )
    assert_allclose(
        result_df['SMMA'].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_smma_polars_without_output_col(df_random_walk: pl.DataFrame) -> None:
    """Test default output column naming."""
    length = 7
    result_df = smma_polars(df_random_walk, close_col='close', length=length)
    expected_col = f'SMMA_{length}'
    assert expected_col in result_df.columns
    close_arr = df_random_walk['close'].to_numpy()
    expected = _smma_reference(close_arr, length)
    assert_allclose(
        result_df[expected_col].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_smma_consistency_with_alligator_import() -> None:
    """Core output matches expected shape and warmup behaviour."""
    close = np.random.randn(100).astype(np.float64)
    length = 10
    result = _smma_numba_core(close, length)
    assert result.shape == close.shape
    assert not np.isnan(result[length - 1:]).any()


# -----------------------------------------------------------------------------
# Validation and NaN-policy tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_smma_numba_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match='must be >= 1'):
        smma_numba(close, length=0)


@pytest.mark.overlap
def test_smma_numba_nan_policy_raise() -> None:
    """Input with NaN and default nan_policy='raise' raises ValueError."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match='NaN'):
        smma_numba(data, length=3)


@pytest.mark.overlap
def test_smma_numba_nan_policy_ffill() -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    result = smma_numba(data, length=3, nan_policy='ffill')
    # after warmup all values must be finite
    assert np.isfinite(result[2:]).all()


@pytest.mark.overlap
def test_smma_numba_with_nan(prices_with_nan):
    """NaN in input poisons the SMMA recurrence permanently."""
    length = 5
    result = smma_numba(prices_with_nan, length=length, nan_policy='ignore')
    # NaN at index 5 -> initial SMA (indices 0-4) is clean,
    # but from index 5 the recurrence propagates NaN forever.
    assert np.isfinite(result[4])
    assert np.isnan(result[5:]).all()


@pytest.mark.overlap
def test_smma_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    length = 5
    result = smma_numba(prices_with_inf, length=length, nan_policy='ignore')
    assert np.isfinite(result[4])
    assert np.isnan(result[5:]).all()


@pytest.mark.overlap
def test_smma_numba_empty(prices_empty):
    """Empty input returns empty array."""
    result = smma_numba(prices_empty, length=5)
    assert result.size == 0


@pytest.mark.overlap
def test_smma_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = smma_numba(prices_all_nan, length=5, nan_policy='ignore')
    assert np.isnan(result).all()
    result_fill = smma_numba(
        prices_all_nan, length=5, fillna=0.0, nan_policy='ignore'
    )
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_smma_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    result = smma_numba(prices_extreme, length=5, nan_policy='ignore')
    assert result is not None