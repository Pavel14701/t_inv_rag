# -*- coding: utf-8 -*-
"""Unit tests for Bollinger Bands (BBANDS) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...volatility.bbands import bbands_numpy, bbands, bbands_polars
from ...overlap.sma import sma_ind
from ...statistics.stdev import stdev_ind


@pytest.mark.statistics
def test_bbands_numpy_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test bbands_numpy shapes, warm-up NaN and band ordering."""
    close = prices_random_walk
    length = 20
    lower, mid, upper, bandwidth, percent_b = bbands_numpy(
        close, length=length, use_talib=False,
    )

    assert lower.shape == mid.shape == upper.shape == close.shape
    assert bandwidth.shape == percent_b.shape == close.shape
    # SMA warm-up: first length-1 values are NaN.
    assert np.isnan(mid[: length - 1]).all()
    assert np.isfinite(lower[length - 1:]).all()
    assert np.isfinite(upper[length - 1:]).all()
    assert np.isfinite(bandwidth[length - 1:]).all()
    # Bands bracket the mid line.
    valid = slice(length - 1, len(close))
    assert (lower[valid] <= mid[valid]).all()
    assert (mid[valid] <= upper[valid]).all()


@pytest.mark.statistics
def test_bbands_numpy_formula_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Bands match mid +- k * stdev with mid = SMA(close)."""
    close = prices_random_walk
    length = 20
    lower, mid, upper, _, _ = bbands_numpy(
        close, length=length, lower_std=2.0, upper_std=2.0, use_talib=False,
    )
    expected_mid = sma_ind(close, length=length, use_talib=False)
    expected_std = stdev_ind(close, length=length, ddof=1, use_talib=False)
    assert_allclose(mid, expected_mid, rtol=1e-12, equal_nan=True)
    assert_allclose(
        lower, expected_mid - 2.0 * expected_std, rtol=1e-9, equal_nan=True,
    )
    assert_allclose(
        upper, expected_mid + 2.0 * expected_std, rtol=1e-9, equal_nan=True,
    )


@pytest.mark.statistics
def test_bbands_numpy_constant_close() -> None:
    """Constant window: zero deviation -> bandwidth 0 and %B 0/0 = NaN.

    The 0/0 must be computed silently (errstate) and yield NaN, never a
    RuntimeWarning, inf or junk.
    """
    import warnings

    close = np.full(30, 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        lower, mid, upper, bandwidth, percent_b = bbands_numpy(
            close, length=5, use_talib=False,
        )
    assert (lower[4:] == 5.0).all()
    assert (upper[4:] == 5.0).all()
    assert (bandwidth[4:] == 0.0).all()
    assert np.isnan(percent_b[4:]).all()


@pytest.mark.statistics
def test_bbands_numpy_asymmetric_multipliers(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Different lower/upper multipliers widen each band independently."""
    close = prices_random_walk
    lower, mid, upper, _, _ = bbands_numpy(
        close, length=20, lower_std=1.0, upper_std=3.0, use_talib=False,
    )
    valid = ~np.isnan(mid)
    diff_up = (upper[valid] - mid[valid]) / 3.0
    assert_allclose(mid[valid] - lower[valid], diff_up, rtol=1e-9)


@pytest.mark.statistics
def test_bbands_numpy_length_too_short_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 is rejected."""
    with pytest.raises(ValueError, match='length must be >= 1'):
        bbands_numpy(prices_random_walk, length=0, use_talib=False)


@pytest.mark.statistics
def test_bbands_numpy_negative_std_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Negative band multipliers would flip the bands and are rejected."""
    with pytest.raises(ValueError, match='std multipliers must be >= 0'):
        bbands_numpy(
            prices_random_walk, length=20, lower_std=-1.0, use_talib=False,
        )


@pytest.mark.statistics
def test_bbands_numpy_nan_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN input raises via the underlying SMA (nan_policy='raise')."""
    close = prices_random_walk.copy()
    close[3] = np.nan
    with pytest.raises(ValueError, match='Input contains NaN'):
        bbands_numpy(close, length=20, use_talib=False)


@pytest.mark.statistics
def test_bbands_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna propagate to all five series."""
    close = prices_random_walk
    lower, mid, upper, bandwidth, percent_b = bbands_numpy(
        close, length=20, offset=1, fillna=0.0, use_talib=False,
    )
    assert lower[0] == 0.0
    assert mid[0] == 0.0
    assert upper[0] == 0.0
    assert bandwidth[0] == 0.0
    assert percent_b[0] == 0.0


@pytest.mark.statistics
def test_bbands_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test bbands with Polars Series input."""
    result = bbands(pl.Series(prices_random_walk), length=20, use_talib=False)
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert np.isfinite(result[0][19:]).all()


@pytest.mark.statistics
def test_bbands_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test bbands_polars adds the five default columns."""
    result_df = bbands_polars(df_random_walk, length=20, use_talib=False)
    for col in (
        'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0',
        'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0',
    ):
        assert col in result_df.columns
        assert result_df[col].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)