# -*- coding: utf-8 -*-
"""Unit tests for Chande Momentum Oscillator (CMO) module.

Covers:
- textbook formula parity (rolling sums of up/down changes)
- NaN warm-up pattern (length + drift - 1)
- constant input (denominator zero) -> NaN, no warnings
- non-contiguous and read-only inputs
- parameter validation (length, drift)
- backend divergence note: talib CMO uses Wilder smoothing, so strict
  backend parity is NOT asserted
- offset / fillna semantics
- cmo_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.momentum.cmo import cmo_ind, cmo_numpy, cmo_polars


def _cmo_reference(
    close: npt.NDArray[np.float64],
    length: int,
    drift: int,
) -> np.ndarray:
    """Textbook CMO: (sum_up - sum_down) / (sum_up + sum_down) * 100."""
    n = len(close)
    diff = np.zeros(n)
    diff[drift:] = close[drift:] - close[:-drift]
    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)
    out = np.full(n, np.nan)
    for i in range(length + drift - 1, n):
        sl = slice(i - length + 1, i + 1)
        sp, sn = pos[sl].sum(), neg[sl].sum()
        out[i] = 100.0 * (sp - sn) / (sp + sn) if sp + sn != 0 else np.nan
    return out


@pytest.mark.momentum
def test_cmo_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy backend equals the textbook rolling-sum CMO."""
    for drift in (1, 2):
        result = cmo_numpy(
            prices_random_walk,
            length=14,
            drift=drift,
            use_talib=False,
        )
        expected = _cmo_reference(prices_random_walk, 14, drift)
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_cmo_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN for the first length + drift - 1 positions, finite after."""
    result = cmo_numpy(prices_random_walk, length=14, drift=1, use_talib=False)
    assert np.isnan(result[:14]).all()
    assert np.isfinite(result[14:]).all()


@pytest.mark.momentum
def test_cmo_constant_input_no_warning() -> None:
    """Constant prices -> denominator 0 -> NaN without warnings."""
    flat = np.full(50, 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = cmo_numpy(flat, length=14, use_talib=False)
    assert np.isnan(result[14:]).all()


@pytest.mark.momentum
def test_cmo_non_contiguous_and_readonly(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views and read-only polars buffers must not change results."""
    expected = cmo_numpy(prices_random_walk, use_talib=False)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    close_nc = mk_nc(prices_random_walk)
    assert not close_nc.flags.c_contiguous
    assert_allclose(
        cmo_numpy(close_nc, use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        cmo_ind(pl.Series(prices_random_walk), use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_cmo_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 and drift < 1 are rejected."""
    with pytest.raises(ValueError, match="length must be >= 1"):
        cmo_numpy(prices_random_walk, length=0, use_talib=False)
    with pytest.raises(ValueError, match="drift must be >= 1"):
        cmo_numpy(prices_random_walk, drift=0, use_talib=False)


@pytest.mark.momentum
def test_cmo_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    length = 14
    r0 = cmo_numpy(prices_random_walk, length=length, use_talib=False)
    r2 = cmo_numpy(
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
def test_cmo_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """cmo_polars default column is CMO_{length}."""
    result = cmo_polars(df_ohlc, use_talib=False)
    assert "CMO_14" in result.columns
    assert result["CMO_14"].dtype == pl.Float64
    expected = cmo_numpy(df_ohlc["close"].to_numpy(), use_talib=False)
    assert_allclose(result["CMO_14"].to_numpy(), expected, equal_nan=True)
