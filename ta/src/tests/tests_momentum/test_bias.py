# -*- coding: utf-8 -*-
"""Unit tests for Bias (BIAS) module.

Covers:
- formula parity close / MA - 1 (sma and ema modes)
- warm-up NaNs follow the MA warm-up
- ma == 0 -> IEEE inf/NaN without RuntimeWarning
- parameter validation (length)
- offset / fillna semantics
- bias_ind with pl.Series / read-only buffers
- bias_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...ma import ma_mode
from ...momentum.bias import bias_ind, bias_numpy, bias_polars
from ...overlap.sma import sma_ind


@pytest.mark.momentum
def test_bias_matches_formula_sma(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Bias equals close / SMA(close, length) - 1 (exact)."""
    close = prices_random_walk
    length = 20
    result = bias_numpy(close, length=length, mamode='sma', use_talib=False)
    ma = cast_np(sma_ind(close, length=length, use_talib=False))
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = (close / ma) - 1.0
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


def cast_np(arr: object) -> np.ndarray:
    """Narrow the ma/sma return to np.ndarray for the reference math."""
    assert isinstance(arr, np.ndarray)
    return arr


@pytest.mark.momentum
def test_bias_ema_mode_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """mamode='ema' uses the same ma_mode backend."""
    close = prices_random_walk
    result = bias_numpy(close, length=10, mamode='ema', use_talib=False)
    ma = cast_np(ma_mode('ema', close, length=10, offset=0, fillna=None))
    expected = (close / ma) - 1.0
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_bias_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN exactly on the MA warm-up region."""
    close = prices_random_walk
    length = 20
    result = bias_numpy(close, length=length, mamode='sma', use_talib=False)
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.momentum
def test_bias_zero_prices_no_warnings() -> None:
    """Ma == 0 follows IEEE rules silently (NaN here), no RuntimeWarning."""
    zeros = np.zeros(40)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = bias_numpy(zeros, length=5, mamode='sma', use_talib=False)
    assert np.isnan(result[4:]).all()
    # a single positive price inside an all-zero window gives finite bias
    data = np.zeros(40)
    data[20] = 10.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result2 = bias_numpy(data, length=5, mamode='sma', use_talib=False)
    assert np.isfinite(result2[20:24]).all()


@pytest.mark.momentum
def test_bias_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 is rejected."""
    with pytest.raises(ValueError, match='length must be >= 1'):
        bias_numpy(prices_random_walk, length=0, use_talib=False)


@pytest.mark.momentum
def test_bias_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    close = prices_random_walk
    length = 10
    r0 = bias_numpy(close, length=length, use_talib=False)
    r2 = bias_numpy(
        close, length=length, offset=2, fillna=0.0, use_talib=False,
    )
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(
        r2[length + 1 :], r0[length - 1 : -2], rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_bias_ind_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """bias_ind accepts a read-only polars Series buffer."""
    close = prices_random_walk
    expected = bias_numpy(close, length=10, use_talib=False)
    result = bias_ind(pl.Series(close), length=10, use_talib=False)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_bias_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """bias_polars default column is BIAS_{mamode}_{length}."""
    result = bias_polars(df_ohlc, length=26, use_talib=False)
    assert 'BIAS_sma_26' in result.columns
    assert result['BIAS_sma_26'].dtype == pl.Float64
    assert len(result) == len(df_ohlc)
    expected = bias_numpy(
        df_ohlc['close'].to_numpy(), length=26, use_talib=False,
    )
    assert_allclose(result['BIAS_sma_26'].to_numpy(), expected, equal_nan=True)