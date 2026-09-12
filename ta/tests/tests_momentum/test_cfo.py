# -*- coding: utf-8 -*-
"""Unit tests for Chande Forecast Oscillator (CFO) module.

Covers:
- formula parity scalar * (close - tsf) / close
- backend parity (numpy vs talib)
- warm-up NaNs (length - 1)
- zero close prices must not emit RuntimeWarnings
- parameter validation (length)
- offset / fillna semantics
- cfo_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.momentum.cfo import cfo_ind, cfo_numpy, cfo_polars
from ta.src.overlap.linreg import linreg_ind


@pytest.mark.momentum
def test_cfo_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """CFO equals scalar * (close - tsf) / close."""
    length, scalar = 9, 100.0
    close = prices_random_walk
    result = cfo_numpy(close, length=length, scalar=scalar, use_talib=False)
    tsf = np.asarray(
        linreg_ind(
            close,
            length=length,
            mode="tsf",
            offset=0,
            fillna=None,
            use_talib=False,
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = scalar * (close - tsf) / close
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_cfo_backend_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy and talib backends agree on the finite tail."""
    a = cfo_numpy(prices_random_walk, length=9, use_talib=False)
    b = cfo_numpy(prices_random_walk, length=9, use_talib=True)
    assert_allclose(a[8:], b[8:], rtol=1e-8, equal_nan=True)


@pytest.mark.momentum
def test_cfo_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN for the first length - 1 positions, finite after."""
    result = cfo_numpy(prices_random_walk, length=9, use_talib=False)
    assert np.isnan(result[:8]).all()
    assert np.isfinite(result[8:]).all()


@pytest.mark.momentum
def test_cfo_zero_close_no_warning() -> None:
    """Zero close prices must not raise RuntimeWarnings."""
    close = np.linspace(1.0, 10.0, 50)
    close[20:] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = cfo_numpy(close, length=9, use_talib=False)
    # close == 0 -> (close - tsf) / close is inf or nan, never a warning
    assert not np.isfinite(result[20:]).any()
    assert np.isfinite(result[8:20]).all()


@pytest.mark.momentum
def test_cfo_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 is rejected."""
    with pytest.raises(ValueError, match="length must be >= 1"):
        cfo_numpy(prices_random_walk, length=0, use_talib=False)


@pytest.mark.momentum
def test_cfo_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    length = 9
    r0 = cfo_numpy(prices_random_walk, length=length, use_talib=False)
    r2 = cfo_numpy(
        prices_random_walk,
        length=length,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(
        r2[length + 1 :],
        r0[length - 1 : -2],
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_cfo_ind_accepts_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """cfo_ind accepts numpy arrays and polars Series alike."""
    expected = cfo_numpy(prices_random_walk, use_talib=False)
    assert_allclose(
        cfo_ind(pl.Series(prices_random_walk), use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_cfo_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """cfo_polars default column is CFO_{length}."""
    result = cfo_polars(df_ohlc, use_talib=False)
    assert "CFO_9" in result.columns
    assert result["CFO_9"].dtype == pl.Float64
    expected = cfo_numpy(df_ohlc["close"].to_numpy(), use_talib=False)
    assert_allclose(result["CFO_9"].to_numpy(), expected, equal_nan=True)
