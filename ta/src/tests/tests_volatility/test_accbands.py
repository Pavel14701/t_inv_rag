# -*- coding: utf-8 -*-
"""Unit tests for Acceleration Bands (ACCBANDS) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...volatility.accbands import accbands_numpy, accbands, accbands_polars
from ...overlap.sma import sma_ind


def _ohlc_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 11,
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
def test_accbands_numpy_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test accbands_numpy shapes, warm-up NaN and band ordering."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    length = 20
    upper, mid, lower = accbands_numpy(
        high, low, close, length=length, use_talib=False,
    )

    assert upper.shape == mid.shape == lower.shape == close.shape
    # SMA warm-up: first length-1 values are NaN.
    assert np.isnan(mid[: length - 1]).all()
    assert np.isfinite(upper[length - 1:]).all()
    assert np.isfinite(mid[length - 1:]).all()
    assert np.isfinite(lower[length - 1:]).all()
    # Positive price ranges widen bands: upper above mid above lower.
    valid = slice(length - 1, len(close))
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


@pytest.mark.statistics
def test_accbands_numpy_mid_is_sma(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """The middle band is exactly the SMA of close."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    _, mid, _ = accbands_numpy(high, low, close, length=20, use_talib=False)
    expected = sma_ind(close, length=20, use_talib=False)
    assert_allclose(mid, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.statistics
def test_accbands_numpy_non_contiguous_input(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Non-contiguous views must give identical results to copies.

    Regression: the old `for arr in (...): arr = ascontiguousarray(arr)`
    loop rebound the loop variable and silently passed strided arrays on
    to the numba backends.
    """
    high, low, close = _ohlc_arrays(prices_random_walk)
    # Strided (non-contiguous) views with the same values.
    high_nc = np.stack([high, high], axis=1)[:, 0]
    low_nc = np.stack([low, low], axis=1)[:, 0]
    close_nc = np.stack([close, close], axis=1)[:, 0]
    assert not high_nc.flags.c_contiguous

    expected = accbands_numpy(
        np.ascontiguousarray(high),
        np.ascontiguousarray(low),
        np.ascontiguousarray(close),
        length=10, use_talib=False,
    )
    result = accbands_numpy(
        high_nc, low_nc, close_nc, length=10, use_talib=False,
    )
    for res, exp in zip(result, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.statistics
def test_accbands_numpy_length_too_short_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 is rejected."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    with pytest.raises(ValueError, match='length must be >= 1'):
        accbands_numpy(high, low, close, length=0, use_talib=False)


@pytest.mark.statistics
def test_accbands_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna propagate to all three bands."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    upper, mid, lower = accbands_numpy(
        high, low, close, length=20, offset=1, fillna=0.0, use_talib=False,
    )
    assert upper[0] == 0.0
    assert mid[0] == 0.0
    assert lower[0] == 0.0


@pytest.mark.statistics
def test_accbands_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test accbands with Polars Series input."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    upper, mid, lower = accbands(
        pl.Series(high), pl.Series(low), pl.Series(close),
        length=20, use_talib=False,
    )
    assert isinstance(upper, np.ndarray)
    assert np.isfinite(upper[19:]).all()


@pytest.mark.statistics
def test_accbands_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """Test accbands_polars adds the three default columns."""
    result_df = accbands_polars(df_ohlc, length=20, use_talib=False)
    for col in ('ACCBU_20', 'ACCBM_20', 'ACCBL_20'):
        assert col in result_df.columns
        assert result_df[col].dtype == pl.Float64
    assert len(result_df) == len(df_ohlc)