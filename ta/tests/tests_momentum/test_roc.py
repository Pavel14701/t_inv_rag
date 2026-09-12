# -*- coding: utf-8 -*-
"""Unit tests for Rate of Change (ROC) module.

Covers:
- formula parity scalar * (close[i] - close[i-length]) / close[i-length]
- backend parity (numpy vs talib), incl. scalar scaling
- warm-up NaNs (length)
- zero denominator -> NaN without warnings
- non-contiguous and read-only inputs
- parameter validation (length)
- offset / fillna semantics
- roc_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.momentum.roc import roc_ind, roc_numpy, roc_polars


@pytest.mark.momentum
def test_roc_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy backend equals the direct ROC formula."""
    length, scalar = 10, 100.0
    close = prices_random_walk
    result = roc_numpy(close, length=length, scalar=scalar, use_talib=False)
    expected = np.full(len(close), np.nan)
    expected[length:] = (
        scalar * (close[length:] - close[:-length]) / close[:-length]
    )
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_roc_backend_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy and talib backends agree on the finite tail."""
    a = roc_numpy(prices_random_walk, length=10, use_talib=False)
    b = roc_numpy(prices_random_walk, length=10, use_talib=True)
    assert_allclose(a[10:], b[10:], rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_roc_scalar_scaling(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """scalar=1.0 gives the fraction, scalar=100.0 the percentage."""
    fraction = roc_numpy(
        prices_random_walk, length=10, scalar=1.0, use_talib=False
    )
    percent = roc_numpy(
        prices_random_walk, length=10, scalar=100.0, use_talib=False
    )
    assert_allclose(fraction[10:], percent[10:] / 100.0, rtol=1e-12)


@pytest.mark.momentum
def test_roc_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN for the first `length` positions, finite after."""
    result = roc_numpy(prices_random_walk, length=10, use_talib=False)
    assert np.isnan(result[:10]).all()
    assert np.isfinite(result[10:]).all()


@pytest.mark.momentum
def test_roc_zero_denominator_no_warning() -> None:
    """Zero base price -> NaN, no RuntimeWarning from the numba core."""
    close = np.linspace(1.0, 10.0, 30)
    close[15:] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = roc_numpy(close, length=10, use_talib=False)
    # Base price hits zero first at i=25 (close[15] == 0).
    assert np.isnan(result[25:]).all()
    assert np.isfinite(result[10:15]).all()


@pytest.mark.momentum
def test_roc_non_contiguous_and_readonly(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views and read-only polars buffers must not change results."""
    expected = roc_numpy(prices_random_walk, use_talib=False)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    close_nc = mk_nc(prices_random_walk)
    assert not close_nc.flags.c_contiguous
    assert_allclose(
        roc_numpy(close_nc, use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        roc_ind(pl.Series(prices_random_walk), use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_roc_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 is rejected."""
    with pytest.raises(ValueError, match="length must be >= 1"):
        roc_numpy(prices_random_walk, length=0, use_talib=False)


@pytest.mark.momentum
def test_roc_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    length = 10
    r0 = roc_numpy(prices_random_walk, length=length, use_talib=False)
    r2 = roc_numpy(
        prices_random_walk,
        length=length,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(
        r2[length + 2 :],
        r0[length:-2],
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_roc_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """roc_polars default column is ROC_{length}."""
    result = roc_polars(df_ohlc, use_talib=False)
    assert "ROC_10" in result.columns
    assert result["ROC_10"].dtype == pl.Float64
    expected = roc_numpy(df_ohlc["close"].to_numpy(), use_talib=False)
    assert_allclose(result["ROC_10"].to_numpy(), expected, equal_nan=True)
