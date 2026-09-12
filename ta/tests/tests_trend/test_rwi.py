# -*- coding: utf-8 -*-
"""Unit tests for Random Walk Index (RWI) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.trend.rwi import rwi_ind, rwi_numpy, rwi_polars
from ta.src.volatility.atr import atr_ind


def _ohlc_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 7,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Derive realistic high/low/close arrays from the price fixture."""
    rng = np.random.default_rng(seed)
    close = prices_random_walk
    noise = np.abs(rng.normal(0.0, 0.5, len(close))) + 0.1
    high = close + noise
    low = close - noise
    return high, low, close


@pytest.mark.statistics
def test_rwi_numpy_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test rwi_numpy shapes and warm-up NaN region."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    length = 5
    rwi_high, rwi_low = rwi_numpy(
        high,
        low,
        close,
        length=length,
        use_talib=False,
    )

    assert rwi_high.shape == close.shape
    assert rwi_low.shape == close.shape
    assert rwi_high.dtype == np.float64
    # First `length` values are NaN (shifted comparison needs i >= length).
    assert np.isnan(rwi_high[:length]).all()
    assert np.isnan(rwi_low[:length]).all()
    assert np.isfinite(rwi_high[length:]).all()
    assert np.isfinite(rwi_low[length:]).all()


@pytest.mark.statistics
def test_rwi_numpy_formula_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Values match the definition (high/low shift over ATR*sqrt(length))."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    length = 5
    rwi_high, rwi_low = rwi_numpy(
        high,
        low,
        close,
        length=length,
        use_talib=False,
    )
    atr = atr_ind(
        high,
        low,
        close,
        length=length,
        mamode="rma",
        use_talib=False,
        nan_policy="ignore",
    )
    denom_factor = np.sqrt(length)
    for i in range(length, len(close), 9):
        denom = atr[i] * denom_factor
        exp_high = (high[i] - low[i - length]) / denom
        exp_low = (high[i - length] - low[i]) / denom
        assert_allclose(rwi_high[i], exp_high, rtol=1e-9)
        assert_allclose(rwi_low[i], exp_low, rtol=1e-9)


@pytest.mark.statistics
def test_rwi_numpy_constant_prices_is_nan() -> None:
    """Zero ATR (constant prices) leaves RWI NaN, never 0/0 junk or inf."""
    n = 30
    close = np.full(n, 5.0)
    high = close + 0.0
    low = close - 0.0
    rwi_high, rwi_low = rwi_numpy(high, low, close, length=5, use_talib=False)
    assert np.isnan(rwi_high).all()
    assert np.isnan(rwi_low).all()
    assert np.isinf(rwi_high).sum() == 0
    assert np.isinf(rwi_low).sum() == 0


@pytest.mark.statistics
def test_rwi_numpy_length_too_short_raises() -> None:
    """Length < 1 is rejected."""
    high = np.array([2.0, 3.0, 4.0])
    low = np.array([1.0, 2.0, 3.0])
    close = np.array([1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="length must be >= 1"):
        rwi_numpy(high, low, close, length=0, use_talib=False)


@pytest.mark.statistics
def test_rwi_numpy_series_too_short_raises() -> None:
    """Fewer than length+1 points cannot produce a single RWI value."""
    n = 5
    close = np.arange(1.0, n + 1.0)
    high = close + 0.5
    low = close - 0.5
    with pytest.raises(ValueError, match="Input series too short"):
        rwi_numpy(high, low, close, length=n, use_talib=False)


@pytest.mark.statistics
def test_rwi_numpy_inf_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Infinite values are rejected up-front (documented contract)."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    high[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        rwi_numpy(high, low, close, length=5, use_talib=False)


@pytest.mark.statistics
def test_rwi_numpy_nan_raises_by_default(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN input raises with nan_policy='raise' (default)."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    low[2] = np.nan
    with pytest.raises(ValueError, match="Input low contains NaN"):
        rwi_numpy(high, low, close, length=5, use_talib=False)


@pytest.mark.statistics
def test_rwi_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test rwi_ind with Polars Series input."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    rwi_high, rwi_low = rwi_ind(
        pl.Series(high),
        pl.Series(low),
        pl.Series(close),
        length=5,
        use_talib=False,
    )
    assert isinstance(rwi_high, np.ndarray)
    assert rwi_high.shape == close.shape
    assert np.isfinite(rwi_high[5:]).all()


@pytest.mark.statistics
def test_rwi_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """Test rwi_polars adds both columns with the default suffix."""
    result_df = rwi_polars(df_ohlc, length=5, use_talib=False)
    assert "RWI_HIGH_5" in result_df.columns
    assert "RWI_LOW_5" in result_df.columns
    assert len(result_df) == len(df_ohlc)
    assert result_df["RWI_HIGH_5"].dtype == pl.Float64
    assert result_df["RWI_HIGH_5"][5:].is_finite().all()
