# -*- coding: utf-8 -*-
"""Unit tests for Exponential Moving Average (EMA) module.

Tests cover:
- Numba core function (_ema_numba_opt)
- ema_numba with offset/fillna, trim, NaN/Inf handling
- ema_talib (if TA-Lib available)
- ema_ind backend selection
- ema_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.ema import (
    _ema_numba_opt,
    ema_numba,
    ema_talib,
    ema_ind,
    ema_polars,
)
from ..._array_ops import _apply_offset_fillna
from ...external import talib_available


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _ema_reference(
    close: npt.NDArray[np.float64],
    length: int
) -> npt.NDArray[np.float64]:
    """Pure Python reference implementation of EMA.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of prices.
    length : int
        EMA period.

    Returns
    -------
    np.ndarray
        EMA array with NaNs for first (length-1) elements.

    """
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


# -----------------------------------------------------------------------------
# Tests for _ema_numba_opt
# -----------------------------------------------------------------------------
def test_ema_numba_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Core Numba EMA matches pure Python reference."""
    close = prices_random_walk
    length = 10
    result = _ema_numba_opt(close, length)
    expected = _ema_reference(close, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


def test_ema_numba_core_short_window() -> None:
    """If window > length, all values are NaN."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 5
    result = _ema_numba_opt(close, length)
    assert np.isnan(result).all()


# -----------------------------------------------------------------------------
# Tests for ema_numba
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_ema_numba_basic(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """ema_numba basic calculation matches reference."""
    close = prices_random_walk
    length = 10
    result = ema_numba(close, length=length)
    expected = _ema_reference(close, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_ema_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test offset shift and fillna using the real _apply_offset_fillna."""
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0
    base = ema_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = ema_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_ema_numba_nan_policy_raise() -> None:
    """'raise' policy raises on NaN."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match='Input close contains NaN values'):
        ema_numba(close, length=3, nan_policy='raise')


@pytest.mark.overlap
def test_ema_numba_trim(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Trimming removes first (length-1) values."""
    close = prices_random_walk
    length = 10
    result = ema_numba(close, length=length, trim=True)
    expected_len = len(close) - (length - 1)
    assert len(result) == expected_len
    full = ema_numba(close, length=length, trim=False)
    assert_allclose(result, full[length - 1:], rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_ema_numba_trim_with_offset(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Trim with offset: both operations applied correctly."""
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0
    base = _ema_reference(close, length)
    base_trimmed = base[length - 1:]
    expected = _apply_offset_fillna(base_trimmed, offset, fillna)
    result = ema_numba(
        close, length=length, offset=offset,
        fillna=fillna, trim=True
    )
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for ema_talib (if available)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
@pytest.mark.overlap
def test_ema_talib_against_reference(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """ema_talib matches reference (within tolerance)."""
    close = prices_random_walk
    length = 10
    result_talib = ema_talib(close, length=length)
    expected = _ema_reference(close, length)
    assert_allclose(
        result_talib, expected, rtol=1e-3,
        atol=1e-3, equal_nan=True
    )


@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
@pytest.mark.overlap
def test_ema_talib_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """ema_talib offset and fillna work correctly."""
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0
    base = ema_talib(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = ema_talib(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
@pytest.mark.overlap
def test_ema_talib_trim(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """ema_talib trim removes first (length-1) values."""
    close = prices_random_walk
    length = 10
    result_talib = ema_talib(close, length=length, trim=True)
    expected_len = len(close) - (length - 1)
    assert len(result_talib) == expected_len
    full = ema_talib(close, length=length, trim=False)
    assert_allclose(result_talib, full[length - 1:], rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for ema_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_ema_ind_uses_numba(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """ema_ind with use_talib=False uses Numba."""
    close = prices_random_walk
    length = 10
    result = ema_ind(close, length=length, use_talib=False)
    expected = _ema_reference(close, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
@pytest.mark.overlap
def test_ema_ind_uses_talib(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """ema_ind with use_talib=True uses TA-Lib."""
    close = prices_random_walk
    length = 10
    result = ema_ind(close, length=length, use_talib=True)
    expected = ema_talib(close, length=length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_ema_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """ema_ind accepts Polars Series."""
    s = pl.Series(prices_random_walk)
    length = 10
    result = ema_ind(s, length=length, use_talib=False)
    expected = _ema_reference(prices_random_walk, length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for ema_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_ema_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """ema_polars adds correct column."""
    length = 10
    result_df = ema_polars(
        df_random_walk,
        close_col='close',
        length=length,
        use_talib=False,
        output_col='EMA'
    )
    assert 'EMA' in result_df.columns
    assert result_df['EMA'].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk['close'].to_numpy()
    expected = _ema_reference(close_arr, length)
    assert_allclose(
        result_df['EMA'].to_numpy(),
        expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_ema_polars_default_output_col() -> None:
    """Default output column name is f'EMA_{length}'."""
    df = pl.DataFrame(
        {'close': [
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0
        ]}
    )
    length = 3
    result_df = ema_polars(
        df, close_col='close',
        length=length, use_talib=False
    )
    expected_col = f'EMA_{length}'
    assert expected_col in result_df.columns


@pytest.mark.overlap
def test_ema_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """ema_polars applies offset and fillna."""
    length = 10
    offset = 3
    fillna = 0.0
    result_df = ema_polars(
        df_random_walk,
        close_col='close',
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=False,
        output_col='EMA'
    )
    close_arr = df_random_walk['close'].to_numpy()
    expected = ema_numba(
        close_arr, length=length,
        offset=offset, fillna=fillna
    )
    assert_allclose(
        result_df['EMA'].to_numpy(), expected,
        rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_ema_numba_with_nan(prices_with_nan):
    """NaN in input propagates correctly through EMA calculation."""  # noqa: D403, E501
    length = 5
    result = ema_numba(prices_with_nan, length=length, nan_policy='ignore')
    # NaN at index 5. EMA with window 5:
    # - indices 0-3: NaN (insufficient data)
    # - index 4: SMA of indices 0-4 (no NaN) -> finite
    # - index 5: EMA(1-5) includes NaN -> NaN
    # - index 6: EMA(2-6) includes NaN -> NaN
    # ... and so on
    assert np.isnan(result[:4]).all()      # 0-3 NaN
    assert np.isfinite(result[4])          # index 4 finite
    assert np.isnan(result[5:]).all()      # from 5 onward all NaN


@pytest.mark.overlap
def test_ema_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    length = 5
    result = ema_numba(prices_with_inf, length=length, nan_policy='ignore')
    # Same as NaN test: Inf at index 5 becomes NaN
    assert np.isnan(result[:4]).all()
    assert np.isfinite(result[4])
    assert np.isnan(result[5:]).all()


@pytest.mark.overlap
def test_ema_numba_empty(prices_empty):
    """Empty input raises ValueError because length is insufficient."""
    with pytest.raises(ValueError, match='Input series too short'):
        ema_numba(prices_empty, length=5)


@pytest.mark.overlap
def test_ema_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = ema_numba(prices_all_nan, length=5, nan_policy='ignore')
    assert np.isnan(result).all()
    result_fill = ema_numba(
        prices_all_nan, length=5,
        fillna=0.0, nan_policy='ignore'
    )
    # _apply_offset_fillna replaces all NaNs with fillna
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_ema_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    length = 5
    result = ema_numba(prices_extreme, length=length, nan_policy='ignore')
    assert result is not None
    # At least some finite values after index length-1
    assert np.isfinite(result[length:]).any()


@pytest.mark.overlap
def test_ema_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly."""
    # Create a copy of the close column with NaN inserted at index 5
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([pl.Series('close', close_arr)])
    result_df = ema_polars(
        df_with_nan,
        close_col='close',
        length=5,
        output_col='EMA',
        nan_policy='ignore'
    )
    assert 'EMA' in result_df.columns
    assert len(result_df) == len(df_random_walk)
    ema_vals = result_df['EMA'].to_numpy()
    # After NaN appears, EMA should remain NaN forever
    assert np.isnan(ema_vals[5:]).all()
