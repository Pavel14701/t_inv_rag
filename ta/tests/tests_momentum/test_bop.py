# -*- coding: utf-8 -*-
"""Unit tests for Balance of Power (BOP) module.

Covers:
- formula parity (close-open)/(high-low) on both backends
- TA-Lib vs numpy parity
- zero-range bars (high == low): 0.0 on both backends, no RuntimeWarning
- custom scalar multiplier
- non-contiguous and read-only inputs
- offset / fillna semantics
- bop_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...external import talib_available
from ...momentum.bop import bop_ind, bop_numpy, bop_polars, bop_talib


@pytest.fixture
def ohlc_arrays(
    prices_random_walk: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Derive realistic OHLC arrays from the price fixture."""
    rng = np.random.default_rng(5)
    close = prices_random_walk
    open_ = np.roll(close, 1) + rng.normal(0.0, 0.1, len(close))
    open_[0] = close[0]
    noise = np.abs(rng.normal(0.0, 0.5, len(close))) + 0.1
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    return open_, high, low, close


@pytest.mark.momentum
def test_bop_numpy_matches_formula(ohlc_arrays) -> None:
    """bop_numpy equals scalar * (close - open) / (high - low)."""
    open_, high, low, close = ohlc_arrays
    result = bop_numpy(open_, high, low, close)
    expected = (close - open_) / (high - low)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_bop_talib_parity(ohlc_arrays) -> None:
    """TA-Lib and numpy backends agree on generic data."""
    open_, high, low, close = ohlc_arrays
    assert_allclose(
        bop_numpy(open_, high, low, close),
        bop_talib(open_, high, low, close),
        rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_bop_zero_range_is_zero_no_warnings() -> None:
    """Zero-range bars give 0.0 (talib semantics) on both backends.

    Regression: the numpy backend used to return NaN/inf with a
    RuntimeWarning, diverging from the TA-Lib backend (0.0).
    """
    n = 40
    zig_o = np.full(n, 5.0)
    zig_h = np.full(n, 5.0)
    zig_l = np.full(n, 5.0)
    zig_c = np.full(n, 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        res_np = bop_numpy(zig_o, zig_h, zig_l, zig_c)
        assert np.isfinite(res_np).all()
        assert (res_np == 0.0).all()
        if talib_available:
            res_tl = bop_talib(zig_o, zig_h, zig_l, zig_c)
            assert (res_tl == 0.0).all()


@pytest.mark.momentum
def test_bop_custom_scalar(ohlc_arrays) -> None:
    """Scalar multiplies the raw ratio on the numpy backend."""
    open_, high, low, close = ohlc_arrays
    r1 = bop_numpy(open_, high, low, close, scalar=1.0)
    r2 = bop_numpy(open_, high, low, close, scalar=100.0)
    assert_allclose(r2, 100.0 * r1, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_bop_non_contiguous_and_readonly(ohlc_arrays) -> None:
    """Strided views and read-only polars buffers must not change results."""
    open_, high, low, close = ohlc_arrays
    expected = bop_numpy(open_, high, low, close)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    assert_allclose(
        bop_numpy(
            mk_nc(open_), mk_nc(high), mk_nc(low), mk_nc(close),
        ),
        expected, rtol=1e-12, equal_nan=True,
    )
    assert_allclose(
        bop_ind(
            pl.Series(open_), pl.Series(high),
            pl.Series(low), pl.Series(close),
            use_talib=False,
        ),
        expected, rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_bop_offset_fillna(ohlc_arrays) -> None:
    """Offset shifts and fillna replaces ALL NaN."""
    open_, high, low, close = ohlc_arrays
    r0 = bop_numpy(open_, high, low, close)
    r2 = bop_numpy(open_, high, low, close, offset=2, fillna=0.0)
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(r2[2:], r0[:-2], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_bop_ind_use_talib_switch(ohlc_arrays) -> None:
    """bop_ind honours use_talib=False explicitly."""
    open_, high, low, close = ohlc_arrays
    expected = bop_numpy(open_, high, low, close)
    assert_allclose(
        bop_ind(open_, high, low, close, use_talib=False),
        expected, rtol=1e-12, equal_nan=True,
    )


@pytest.mark.momentum
def test_bop_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """bop_polars returns date + BOP column matching the numpy backend."""
    result = bop_polars(df_ohlc, use_talib=False)
    assert 'BOP' in result.columns
    assert result['BOP'].dtype == pl.Float64
    assert len(result) == len(df_ohlc)
    expected = bop_numpy(
        df_ohlc['open'].to_numpy(), df_ohlc['high'].to_numpy(),
        df_ohlc['low'].to_numpy(), df_ohlc['close'].to_numpy(),
    )
    assert_allclose(result['BOP'].to_numpy(), expected, equal_nan=True)