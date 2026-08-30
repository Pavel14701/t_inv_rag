# -*- coding: utf-8 -*-
"""Unit tests for Z Candles (cdl_z).

Tests cover:
- cdl_z_numpy with rolling and global z-score
- offset and fillna
- universal cdl_z with Polars Series
- cdl_z_polars with default and custom suffix
- cdl_z_polars with full=True
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ....candle.cdl_z import cdl_z_numpy, cdl_z, cdl_z_polars


# -----------------------------------------------------------------------------
# Tests with numpy arrays (using fixed data for reproducibility)
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_cdl_z_numpy_basic() -> None:
    """Test Z Candles with default parameters (rolling z-score)."""
    np.random.seed(42)
    n = 100
    open_ = np.random.randn(n) * 10 + 100
    high = open_ + np.abs(np.random.randn(n) * 2)
    low = open_ - np.abs(np.random.randn(n) * 2)
    close = open_ + np.random.randn(n) * 1

    result = cdl_z_numpy(open_, high, low, close, length=30, ddof=1, full=False, use_talib=False)

    expected_keys = ['open_Z_30_1', 'high_Z_30_1', 'low_Z_30_1', 'close_Z_30_1']
    assert all(key in result for key in expected_keys)
    for key in expected_keys:
        assert result[key].shape == (n,)
        assert result[key].dtype == np.float64
    assert np.any(result['open_Z_30_1'] != 0.0)


@pytest.mark.statistics
def test_cdl_z_numpy_full() -> None:
    """Test Z Candles with full=True (global z-score)."""
    np.random.seed(42)
    n = 100
    open_ = np.random.randn(n) * 10 + 100
    high = open_ + np.abs(np.random.randn(n) * 2)
    low = open_ - np.abs(np.random.randn(n) * 2)
    close = open_ + np.random.randn(n) * 1

    result = cdl_z_numpy(open_, high, low, close, full=True, ddof=1, use_talib=False)

    expected_keys = ['open_Za', 'high_Za', 'low_Za', 'close_Za']
    assert all(key in result for key in expected_keys)
    for key in expected_keys:
        arr = result[key]
        assert abs(arr.mean()) < 0.1
        assert 0.8 < arr.std(ddof=1) < 1.2


@pytest.mark.statistics
def test_cdl_z_offset_fillna() -> None:
    """Test Z Candles with offset and fillna."""
    np.random.seed(42)
    n = 100
    open_ = np.random.randn(n) * 10 + 100
    high = open_ + np.abs(np.random.randn(n) * 2)
    low = open_ - np.abs(np.random.randn(n) * 2)
    close = open_ + np.random.randn(n) * 1

    result_no_offset = cdl_z_numpy(open_, high, low, close, length=30, ddof=1, full=False, use_talib=False)
    result_offset = cdl_z_numpy(open_, high, low, close, length=30, ddof=1, full=False,
                                offset=1, fillna=0.0, use_talib=False)

    for key in result_offset:
        assert result_offset[key].shape == (n,)
        assert result_offset[key][0] == 0.0
        assert_allclose(result_offset[key][1:], result_no_offset[key][:-1], rtol=1e-6)


# -----------------------------------------------------------------------------
# Polars integration tests (using shared fixtures)
# -----------------------------------------------------------------------------

@pytest.mark.statistics
def test_cdl_z_universal_with_polars_series(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Test universal cdl_z with Polars Series."""
    n = len(prices_random_walk)
    # Create OHLC from random walk
    open_s = pl.Series(prices_random_walk + np.random.randn(n) * 0.5)
    high_s = pl.Series(prices_random_walk + np.abs(np.random.randn(n) * 0.8))
    low_s = pl.Series(prices_random_walk - np.abs(np.random.randn(n) * 0.8))
    close_s = pl.Series(prices_random_walk)

    result = cdl_z(open_s, high_s, low_s, close_s, length=30, ddof=1, full=False, use_talib=False)

    expected_keys = ['open_Z_30_1', 'high_Z_30_1', 'low_Z_30_1', 'close_Z_30_1']
    assert all(key in result for key in expected_keys)
    for key in expected_keys:
        assert isinstance(result[key], np.ndarray)
        assert result[key].shape == (n,)


@pytest.mark.statistics
def test_cdl_z_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """Test cdl_z_polars adds correct columns."""
    result = cdl_z_polars(df_ohlc, length=30, ddof=1, full=False, use_talib=False)

    expected_cols = ['date', 'open_Z_30_1', 'high_Z_30_1', 'low_Z_30_1', 'close_Z_30_1']
    assert list(result.columns) == expected_cols
    assert len(result) == len(df_ohlc)
    for col in expected_cols[1:]:
        assert result[col].dtype == pl.Float64


@pytest.mark.statistics
def test_cdl_z_polars_with_suffix(df_ohlc: pl.DataFrame) -> None:
    """Test cdl_z_polars with custom suffix."""
    result = cdl_z_polars(df_ohlc, length=30, ddof=1, full=False, suffix='custom', use_talib=False)

    expected_cols = ['date', 'open_Zcustom', 'high_Zcustom', 'low_Zcustom', 'close_Zcustom']
    assert list(result.columns) == expected_cols
    for col in expected_cols[1:]:
        assert result[col].dtype == pl.Float64


@pytest.mark.statistics
def test_cdl_z_polars_full(df_ohlc: pl.DataFrame) -> None:
    """Test cdl_z_polars with full=True."""
    result = cdl_z_polars(df_ohlc, full=True, ddof=1, use_talib=False)

    expected_cols = ['date', 'open_Za', 'high_Za', 'low_Za', 'close_Za']
    assert list(result.columns) == expected_cols
    for col in expected_cols[1:]:
        arr = result[col].to_numpy()
        assert abs(arr.mean()) < 0.1
        assert 0.8 < arr.std(ddof=1) < 1.2