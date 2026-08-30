# -*- coding: utf-8 -*-
"""Unit tests for Hull Moving Average (HMA) module.

Tests cover:
- hma_numba against reference implementation
- offset and fillna
- hma_ind with Polars Series
- hma_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
- Different mamode options (sma, ema, wma)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.hma import hma_numba, hma_ind, hma_polars
from ...overlap.sma import sma_ind
from ...overlap.ema import ema_ind
from ...overlap.wma import wma_ind
from ..._array_ops import _apply_offset_fillna


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _hma_reference(
    close: npt.NDArray[np.float64],
    length: int,
    mamode: str = 'wma',
) -> npt.NDArray[np.float64]:
    """Pure Python reference HMA using specified MA."""
    if mamode == 'sma':
        ma_func = sma_ind
    elif mamode == 'ema':
        ma_func = ema_ind
    elif mamode == 'wma':
        ma_func = lambda x, length, **kwargs: wma_ind(  # noqa: E731
            x, length, asc=True, use_talib=False, nan_policy='ignore', **kwargs
        )
    else:
        raise ValueError(f'Unsupported mamode: {mamode}')

    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))

    maf = ma_func(close, half_length)
    mas = ma_func(close, length)
    diff = 2.0 * maf - mas
    return ma_func(diff, sqrt_length)


# -----------------------------------------------------------------------------
# Tests for hma_numba
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hma_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test hma_numba against pure Python reference for all mamodes."""
    close = prices_random_walk
    length = 10

    for mamode in ('sma', 'ema', 'wma'):
        result = hma_numba(close, length=length, mamode=mamode)
        expected = _hma_reference(close, length, mamode)
        assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hma_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using _apply_offset_fillna."""
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0

    base = hma_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)

    result = hma_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hma_numba_short_window() -> None:
    """Window longer than data returns all NaN."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 10
    result = hma_numba(close, length=length)
    assert np.isnan(result).all()


@pytest.mark.overlap
def test_hma_numba_invalid_mamode() -> None:
    """Unsupported mamode raises ValueError."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match='Unsupported mamode'):
        hma_numba(close, length=5, mamode='invalid')


# -----------------------------------------------------------------------------
# Tests for hma_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """hma_ind should accept Polars Series and return NumPy array."""
    s = pl.Series(prices_random_walk)
    length = 10
    result = hma_ind(s, length=length, mamode='wma')
    expected = _hma_reference(prices_random_walk, length, 'wma')
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hma_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """hma_polars should add HMA column correctly."""
    length = 10
    result_df = hma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        output_col='HMA',
    )
    assert 'HMA' in result_df.columns
    assert result_df['HMA'].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)

    close_arr = df_random_walk['close'].to_numpy()
    expected = _hma_reference(close_arr, length, 'wma')
    assert_allclose(result_df['HMA'].to_numpy(), expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hma_polars_default_output_col() -> None:
    """Default output column name is f'HMA_{length}'."""
    df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    length = 3
    result_df = hma_polars(df, close_col='close', length=length)
    expected_col = f'HMA_{length}'
    assert expected_col in result_df.columns


@pytest.mark.overlap
def test_hma_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """hma_polars should apply offset and fillna."""
    length = 10
    offset = 2
    fillna = 0.0
    result_df = hma_polars(
        df_random_walk,
        close_col='close',
        length=length,
        offset=offset,
        fillna=fillna,
        output_col='HMA',
    )
    close_arr = df_random_walk['close'].to_numpy()
    expected = hma_numba(close_arr, length=length, offset=offset, fillna=fillna)
    assert_allclose(result_df['HMA'].to_numpy(), expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
def test_hma_numba_with_nan(prices_with_nan):
    """NaN in input propagates correctly through HMA."""
    length = 5
    result = hma_numba(prices_with_nan, length=length, mamode='wma')
    # NaN at index 5. With WMA, NaN propagates to windows containing it.
    # For length=5, half_length=2, sqrt_length=2.
    # NaN will affect indices where any MA includes NaN.
    # We check that NaN appears somewhere after index 5, but not necessarily all.
    assert np.isnan(result[5:10]).any()
    # After index 10, no NaN should remain (since windows no longer include index 5)
    assert np.isfinite(result[10:]).all()


def test_hma_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so behaves like NaN."""
    length = 5
    result = hma_numba(prices_with_inf, length=length, mamode='wma')
    # Same as NaN test
    assert np.isnan(result[5:10]).any()
    assert np.isfinite(result[10:]).all()


def test_hma_numba_empty(prices_empty):
    """Empty input returns empty array."""
    result = hma_numba(prices_empty, length=5)
    assert result.size == 0


def test_hma_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = hma_numba(prices_all_nan, length=5, mamode='wma')
    assert np.isnan(result).all()

    result_fill = hma_numba(prices_all_nan, length=5, fillna=0.0, mamode='wma')
    assert (result_fill == 0.0).all()


def test_hma_numba_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    length = 5
    result = hma_numba(prices_extreme, length=length, mamode='wma')
    assert result is not None


def test_hma_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly."""
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([pl.Series('close', close_arr)])
    result_df = hma_polars(df_with_nan, close_col='close', length=5, output_col='HMA')
    assert 'HMA' in result_df.columns
    hma_vals = result_df['HMA'].to_numpy()
    assert np.isnan(hma_vals[5:10]).any()
    assert np.isfinite(hma_vals[10:]).all()