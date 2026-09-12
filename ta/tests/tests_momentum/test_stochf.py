# -*- coding: utf-8 -*-
"""Unit tests for Fast Stochastic (STOCHF) module.

Covers:
- formula parity 100 * (close - LL(k)) / (HH(k) - LL(k)), %D = MA(%K)
- backend parity (numpy vs talib)
- warm-up NaNs (k - 1 for %K, k + d - 2 for %D)
- zero-range bars (high == low) must not emit RuntimeWarnings
- non-contiguous inputs (regression: no-op contiguous loop)
- parameter validation (k, d)
- offset / fillna semantics
- stochf_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _rolling_max_numba, _rolling_min_numba
from ta.src.ma import ma_mode
from ta.src.momentum.stochf import stochf_ind, stochf_numpy, stochf_polars


def _np(arr: object) -> np.ndarray:
    assert isinstance(arr, np.ndarray)
    return arr


def _hl_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 47,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    noise = np.abs(rng.normal(0.0, 0.5, len(prices_random_walk))) + 0.1
    return prices_random_walk + noise, prices_random_walk - noise


@pytest.mark.momentum
def test_stochf_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy backend equals the fast stochastic formula."""
    high, low = _hl_arrays(prices_random_walk)
    close, k, d = prices_random_walk, 14, 3
    stoch_k, stoch_d = stochf_numpy(
        high, low, close, k=k, d=d, use_talib=False
    )
    ll = _rolling_min_numba(low, k)
    hh = _rolling_max_numba(high, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_k = 100.0 * (close - ll) / (hh - ll)
    assert_allclose(stoch_k, expected_k, rtol=1e-12, equal_nan=True)
    expected_d = _np(ma_mode("sma", expected_k, length=d, nan_policy="ignore"))
    assert_allclose(stoch_d, expected_d, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stochf_backend_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy and talib backends agree on the finite tail."""
    high, low = _hl_arrays(prices_random_walk)
    k_np, d_np = stochf_numpy(high, low, prices_random_walk, use_talib=False)
    k_tl, d_tl = stochf_numpy(high, low, prices_random_walk, use_talib=True)
    # talib STOCHF has one extra warm-up bar for %K on this data
    assert_allclose(k_np[15:], k_tl[15:], rtol=1e-9, equal_nan=True)
    assert_allclose(d_np[17:], d_tl[17:], rtol=1e-9, equal_nan=True)


@pytest.mark.momentum
def test_stochf_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """%K NaN for k - 1 bars, %D for k + d - 2 bars."""
    high, low = _hl_arrays(prices_random_walk)
    k, d = 14, 3
    stoch_k, stoch_d = stochf_numpy(
        high, low, prices_random_walk, k=k, d=d, use_talib=False
    )
    assert np.isnan(stoch_k[: k - 1]).all()
    assert np.isfinite(stoch_k[k - 1 :]).all()
    assert np.isnan(stoch_d[: k + d - 2]).all()
    assert np.isfinite(stoch_d[k + d - 2 :]).all()


@pytest.mark.momentum
def test_stochf_zero_range_no_warning() -> None:
    """Flat OHLC segment (high == low) -> NaN without warnings."""
    n = 60
    base = np.linspace(50.0, 60.0, n)
    high = base + 0.5
    low = base - 0.5
    close = base.copy()
    high[30:] = low[30:] = close[30:] = 55.0  # zero-range bars
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stoch_k, _ = stochf_numpy(high, low, close, k=14, d=3, use_talib=False)
    # First all-flat window ends at index 43.
    assert np.isnan(stoch_k[43:]).all()
    assert np.isfinite(stoch_k[13:30]).all()


@pytest.mark.momentum
def test_stochf_non_contiguous(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views give the same result (no-op contiguous regression)."""
    high, low = _hl_arrays(prices_random_walk)
    expected = stochf_numpy(high, low, prices_random_walk, use_talib=False)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    assert_allclose(
        stochf_numpy(
            mk_nc(high),
            mk_nc(low),
            mk_nc(prices_random_walk),
            use_talib=False,
        ),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        stochf_ind(
            pl.Series(high),
            pl.Series(low),
            pl.Series(prices_random_walk),
            use_talib=False,
        ),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_stochf_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """K < 1 and d < 1 are rejected."""
    high, low = _hl_arrays(prices_random_walk)
    with pytest.raises(ValueError, match="k must be >= 1"):
        stochf_numpy(high, low, prices_random_walk, k=0, use_talib=False)
    with pytest.raises(ValueError, match="d must be >= 1"):
        stochf_numpy(high, low, prices_random_walk, d=0, use_talib=False)


@pytest.mark.momentum
def test_stochf_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    high, low = _hl_arrays(prices_random_walk)
    k = 14
    k0, d0 = stochf_numpy(high, low, prices_random_walk, k=k, use_talib=False)
    k2, d2 = stochf_numpy(
        high,
        low,
        prices_random_walk,
        k=k,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert k2[0] == 0.0 and k2[1] == 0.0
    assert d2[0] == 0.0
    # %K finite from 13, %D from 15 -> shifted comparison starts at 15/17
    assert_allclose(k2[15:], k0[13:-2], rtol=1e-12, equal_nan=True)
    assert_allclose(d2[17:], d0[15:-2], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stochf_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """stochf_polars default columns are STOCHFk/STOCHFd_{k}_{d}."""
    result = stochf_polars(df_ohlc, use_talib=False)
    assert "STOCHFk_14_3" in result.columns
    assert "STOCHFd_14_3" in result.columns
    assert result["STOCHFk_14_3"].dtype == pl.Float64
    expected_k, expected_d = stochf_numpy(
        df_ohlc["high"].to_numpy(),
        df_ohlc["low"].to_numpy(),
        df_ohlc["close"].to_numpy(),
        use_talib=False,
    )
    assert_allclose(
        result["STOCHFk_14_3"].to_numpy(), expected_k, equal_nan=True
    )
    assert_allclose(
        result["STOCHFd_14_3"].to_numpy(), expected_d, equal_nan=True
    )
