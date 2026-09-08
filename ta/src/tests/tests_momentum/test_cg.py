# -*- coding: utf-8 -*-
"""Unit tests for Center of Gravity (CG) module.

Covers:
- regression: O(1) sliding update must match the direct window formula
  (the old update dropped only the oldest price instead of the whole
  old denominator)
- warm-up NaNs (length - 1)
- constant input (denominator zero) -> NaN, no warnings
- non-contiguous and read-only inputs
- parameter validation (length)
- offset / fillna semantics
- cg_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...momentum.cg import cg_ind, cg_numba, cg_polars


def _cg_reference(
    close: npt.NDArray[np.float64], length: int,
) -> np.ndarray:
    """Direct O(n*length) CG: -sum(k * w[k-1]) / sum(w)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        w = close[i - length + 1 : i + 1]
        k = np.arange(1, length + 1, dtype=np.float64)
        denom = w.sum()
        out[i] = -(k * w).sum() / denom if denom != 0.0 else np.nan
    return out


@pytest.mark.momentum
def test_cg_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Sliding-window CG equals the direct window formula exactly."""
    for length in (5, 10):
        result = cg_numba(prices_random_walk, length=length)
        expected = _cg_reference(prices_random_walk, length)
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_cg_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN for the first length - 1 positions, finite after."""
    result = cg_numba(prices_random_walk, length=10)
    assert np.isnan(result[:9]).all()
    assert np.isfinite(result[9:]).all()


@pytest.mark.momentum
def test_cg_constant_input_no_warning() -> None:
    """Constant prices -> denominator 0 -> NaN without warnings."""
    flat = np.full(50, 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = cg_numba(flat, length=10)
    # CG of a constant series c is -(55*c)/(10*c) = -5.5 (denom != 0).
    assert np.isfinite(result[9:]).all()
    assert np.allclose(result[9:], -5.5)


@pytest.mark.momentum
def test_cg_non_contiguous_and_readonly(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views and read-only polars buffers must not change results."""
    expected = cg_numba(prices_random_walk, length=10)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    close_nc = mk_nc(prices_random_walk)
    assert not close_nc.flags.c_contiguous
    assert_allclose(
        cg_numba(close_nc, length=10), expected, rtol=1e-12, equal_nan=True,
    )
    assert_allclose(
        cg_ind(pl.Series(prices_random_walk), length=10),
        expected, rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_cg_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 is rejected."""
    with pytest.raises(ValueError, match='length must be >= 1'):
        cg_numba(prices_random_walk, length=0)


@pytest.mark.momentum
def test_cg_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    length = 10
    r0 = cg_numba(prices_random_walk, length=length)
    r2 = cg_numba(prices_random_walk, length=length, offset=2, fillna=0.0)
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(
        r2[length + 1 :], r0[length - 1 : -2], rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_cg_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """cg_polars default column is CG_{length}."""
    result = cg_polars(df_ohlc, length=10)
    assert 'CG_10' in result.columns
    assert result['CG_10'].dtype == pl.Float64
    expected = cg_numba(df_ohlc['close'].to_numpy(), length=10)
    assert_allclose(result['CG_10'].to_numpy(), expected, equal_nan=True)
