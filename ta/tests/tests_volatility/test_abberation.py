# -*- coding: utf-8 -*-
"""Unit tests for Aberration (ABERRATION) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.overlap.sma import sma_ind
from ta.src.volatility.abberation import (
    aberration_ind,
    aberration_numpy,
    aberration_polars,
)


def _ohlc_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 13,
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
def test_aberration_numpy_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test aberration_numpy shapes and warm-up regions."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    length, atr_length = 5, 15
    zg, sg, xg, atr = aberration_numpy(
        high,
        low,
        close,
        length=length,
        atr_length=atr_length,
        use_talib=False,
    )

    assert zg.shape == sg.shape == xg.shape == atr.shape == close.shape
    # SMA warm-up of the ZG line.
    assert np.isnan(zg[: length - 1]).all()
    # All lines are finite once both warm-ups are done.
    tail = slice(atr_length + length, len(close))
    assert np.isfinite(zg[tail]).all()
    assert np.isfinite(sg[tail]).all()
    assert np.isfinite(xg[tail]).all()
    assert np.isfinite(atr[tail]).all()


@pytest.mark.statistics
def test_aberration_numpy_band_identities(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """SG and XG are exact zg +- atr shifts of the mid line."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    zg, sg, xg, atr = aberration_numpy(
        high,
        low,
        close,
        length=5,
        atr_length=15,
        use_talib=False,
    )
    assert_allclose(sg, zg + atr, rtol=1e-12, equal_nan=True)
    assert_allclose(xg, zg - atr, rtol=1e-12, equal_nan=True)


@pytest.mark.statistics
def test_aberration_numpy_zg_is_sma_of_hlc3(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """ZG equals the SMA over (high + low + close) / 3."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    zg, _, _, _ = aberration_numpy(
        high,
        low,
        close,
        length=5,
        atr_length=15,
        use_talib=False,
    )
    hlc3 = (high + low + close) / 3.0
    expected = sma_ind(hlc3, length=5, use_talib=False)
    assert_allclose(zg, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.statistics
def test_aberration_numpy_non_contiguous_input(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Non-contiguous views must give identical results to copies.

    Regression: the old `for arr in (...): arr = ascontiguousarray(arr)`
    loop rebound the loop variable and silently passed strided arrays on
    to the numba backends.
    """
    high, low, close = _ohlc_arrays(prices_random_walk)
    high_nc = np.stack([high, high], axis=1)[:, 0]
    low_nc = np.stack([low, low], axis=1)[:, 0]
    close_nc = np.stack([close, close], axis=1)[:, 0]
    assert not high_nc.flags.c_contiguous

    expected = aberration_numpy(
        np.ascontiguousarray(high),
        np.ascontiguousarray(low),
        np.ascontiguousarray(close),
        length=5,
        atr_length=15,
        use_talib=False,
    )
    result = aberration_numpy(
        high_nc,
        low_nc,
        close_nc,
        length=5,
        atr_length=15,
        use_talib=False,
    )
    for res, exp in zip(result, expected, strict=False):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.statistics
def test_aberration_numpy_length_too_short_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 and atr_length < 1 are rejected."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    with pytest.raises(ValueError, match="length must be >= 1"):
        aberration_numpy(high, low, close, length=0, use_talib=False)
    with pytest.raises(ValueError, match="atr_length must be >= 1"):
        aberration_numpy(
            high,
            low,
            close,
            length=5,
            atr_length=0,
            use_talib=False,
        )


@pytest.mark.statistics
def test_aberration_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna propagate to all four lines."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    zg, sg, xg, atr = aberration_numpy(
        high,
        low,
        close,
        length=5,
        atr_length=15,
        offset=1,
        fillna=0.0,
        use_talib=False,
    )
    assert zg[0] == 0.0
    assert sg[0] == 0.0
    assert xg[0] == 0.0
    assert atr[0] == 0.0


@pytest.mark.statistics
def test_aberration_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test aberration_ind with Polars Series input."""
    high, low, close = _ohlc_arrays(prices_random_walk)
    zg, sg, xg, atr = aberration_ind(
        pl.Series(high),
        pl.Series(low),
        pl.Series(close),
        length=5,
        atr_length=15,
        use_talib=False,
    )
    assert isinstance(zg, np.ndarray)
    assert np.isfinite(zg[25:]).all()


@pytest.mark.statistics
def test_aberration_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """Test aberration_polars adds the four default columns."""
    result_df = aberration_polars(
        df_ohlc,
        length=5,
        atr_length=15,
        use_talib=False,
    )
    for col in (
        "ABER_ZG_5_15",
        "ABER_SG_5_15",
        "ABER_XG_5_15",
        "ABER_ATR_5_15",
    ):
        assert col in result_df.columns
        assert result_df[col].dtype == pl.Float64
    assert len(result_df) == len(df_ohlc)
