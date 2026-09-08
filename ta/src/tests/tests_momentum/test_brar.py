# -*- coding: utf-8 -*-
"""Unit tests for BRAR (Balance Ratio / Ability Ratio) module.

Covers:
- formula parity against an independent sliding-window reference (exact)
- warm-up NaNs (window must clear both SMA and drift NaNs)
- zero-denominator guards: flat OHLC -> NaN (never inf), no warnings
- custom scalar multiplier
- non-contiguous and read-only inputs (polars zero-copy regression)
- parameter validation (length, drift)
- offset / fillna semantics
- brar_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...momentum.brar import brar_ind, brar_polars


def _brar_reference(
    open_: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    length: int = 26,
    scalar: float = 100.0,
    drift: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Independent sliding-window reference (window sums)."""
    n = len(close)
    shifted = np.full(n, np.nan)
    shifted[drift:] = close[: n - drift]
    hcy = np.maximum(high - shifted, 0.0)
    cyl = np.maximum(shifted - low, 0.0)

    def _roll(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.full(n, np.nan)
        for i in range(length - 1, n):
            w = x[i - length + 1 : i + 1]
            if np.isnan(w).any():
                out[i] = np.nan
            else:
                out[i] = w.sum()
        return out

    s_ho = _roll(high - open_)
    s_ol = _roll(open_ - low)
    s_hcy = _roll(hcy)
    s_cyl = _roll(cyl)
    with np.errstate(divide='ignore', invalid='ignore'):
        ar = np.where(s_ol != 0.0, scalar * s_ho / s_ol, np.nan)
        br = np.where(s_cyl != 0.0, scalar * s_hcy / s_cyl, np.nan)
    return ar, br


def _ohlc_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 21,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Derive realistic OHLC arrays from the price fixture."""
    rng = np.random.default_rng(seed)
    close = prices_random_walk
    gap = rng.normal(0.0, 0.1, len(close))
    open_ = np.roll(close, 1) + gap
    open_[0] = close[0]
    noise = np.abs(rng.normal(0.0, 0.5, len(close))) + 0.1
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    return open_, high, low, close


@pytest.mark.momentum
def test_brar_matches_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """AR/BR must equal the sliding-window reference exactly."""
    open_, high, low, close = _ohlc_arrays(prices_random_walk)
    length, drift = 26, 1
    ar, br = brar_ind(open_, high, low, close, length=length, drift=drift)
    ref_ar, ref_br = _brar_reference(
        open_, high, low, close, length=length, drift=drift,
    )
    assert_allclose(ar, ref_ar, rtol=1e-12, equal_nan=True)
    assert_allclose(br, ref_br, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_brar_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """AR warms up at length-1; BR also waits out the drift NaN prefix."""
    open_, high, low, close = _ohlc_arrays(prices_random_walk)
    length, drift = 26, 1
    ar, br = brar_ind(open_, high, low, close, length=length, drift=drift)
    assert np.isnan(ar[: length - 1]).all()
    assert np.isfinite(ar[length - 1 :]).all()
    assert np.isnan(br[: length + drift - 1]).all()
    assert np.isfinite(br[length + drift - 1 :]).all()


@pytest.mark.momentum
def test_brar_flat_ohlc_nan_no_warnings() -> None:
    """Flat OHLC -> zero denominators -> NaN, never inf, no warnings."""
    flat = np.full(60, 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ar, br = brar_ind(flat, flat, flat, flat, length=26)
    assert np.isnan(ar).all()
    assert np.isnan(br).all()
    assert not np.isinf(ar).any() and not np.isinf(br).any()


@pytest.mark.momentum
def test_brar_custom_scalar(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Non-default scalar scales both lines linearly."""
    open_, high, low, close = _ohlc_arrays(prices_random_walk)
    ar1, br1 = brar_ind(open_, high, low, close, length=10, scalar=100.0)
    ar2, br2 = brar_ind(open_, high, low, close, length=10, scalar=50.0)
    assert_allclose(ar2, 0.5 * ar1, rtol=1e-12, equal_nan=True)
    assert_allclose(br2, 0.5 * br1, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_brar_non_contiguous_and_readonly(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views and read-only polars buffers must not change results.

    Regression: the old `for arr in (...): arr = ascontiguousarray(arr)`
    loop rebound the loop variable (no-op).
    """
    open_, high, low, close = _ohlc_arrays(prices_random_walk)
    expected = brar_ind(open_, high, low, close, length=10)
    # strided views
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    result_nc = brar_ind(
        mk_nc(open_), mk_nc(high), mk_nc(low), mk_nc(close), length=10,
    )
    for res, exp in zip(result_nc, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    # read-only polars series
    result_pl = brar_ind(
        pl.Series(open_), pl.Series(high),
        pl.Series(low), pl.Series(close), length=10,
    )
    for res, exp in zip(result_pl, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_brar_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 and drift < 1 are rejected."""
    open_, high, low, close = _ohlc_arrays(prices_random_walk)
    with pytest.raises(ValueError, match='length must be >= 1'):
        brar_ind(open_, high, low, close, length=0)
    with pytest.raises(ValueError, match='drift must be >= 1'):
        brar_ind(open_, high, low, close, length=10, drift=0)


@pytest.mark.momentum
def test_brar_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    open_, high, low, close = _ohlc_arrays(prices_random_walk)
    length, drift = 10, 1
    ar0, br0 = brar_ind(open_, high, low, close, length=length)
    ar2, br2 = brar_ind(
        open_, high, low, close, length=length, offset=2, fillna=0.0,
    )
    assert ar2[0] == 0.0 and br2[0] == 0.0
    # compare only past the warm-up zone: fillna replaced warm-up NaNs
    assert_allclose(
        ar2[length + 1 :], ar0[length - 1 : -2], rtol=1e-12, equal_nan=True,
    )
    assert_allclose(
        br2[length + drift + 1 :],
        br0[length + drift - 1 : -2],
        rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_brar_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """brar_polars returns date + AR_{length}/BR_{length} columns."""
    result = brar_polars(df_ohlc, length=26)
    for col in ('AR_26', 'BR_26'):
        assert col in result.columns
        assert result[col].dtype == pl.Float64
    assert len(result) == len(df_ohlc)
    expected = brar_ind(
        df_ohlc['open'].to_numpy(), df_ohlc['high'].to_numpy(),
        df_ohlc['low'].to_numpy(), df_ohlc['close'].to_numpy(), length=26,
    )
    assert_allclose(result['AR_26'].to_numpy(), expected[0], equal_nan=True)
    assert_allclose(result['BR_26'].to_numpy(), expected[1], equal_nan=True)