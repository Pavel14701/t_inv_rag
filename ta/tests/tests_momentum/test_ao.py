# -*- coding: utf-8 -*-
"""Unit tests for Awesome Oscillator (AO) module.

Covers:
- formula parity SMA(median, fast) - SMA(median, slow)
- fast/slow swap symmetry
- warm-up NaNs follow the slow SMA
- non-contiguous and read-only inputs
- parameter validation (fast, slow)
- offset / fillna semantics
- ao_polars DataFrame integration
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.momentum.ao import ao_ind, ao_numpy, ao_polars
from ta.src.overlap.sma import sma_ind


def _np(arr: object) -> np.ndarray:
    """Narrow the sma_ind return to np.ndarray."""
    assert isinstance(arr, np.ndarray)
    return arr


def _hl_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 33,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Derive high/low arrays from the price fixture."""
    rng = np.random.default_rng(seed)
    noise = np.abs(rng.normal(0.0, 0.5, len(prices_random_walk))) + 0.1
    return prices_random_walk + noise, prices_random_walk - noise


@pytest.mark.momentum
def test_ao_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Ao equals SMA((H+L)/2, fast) - SMA((H+L)/2, slow) exactly."""
    high, low = _hl_arrays(prices_random_walk)
    fast, slow = 5, 34
    result = ao_numpy(high, low, fast=fast, slow=slow, use_talib=False)
    median = (high + low) * 0.5
    expected = _np(sma_ind(median, length=fast, use_talib=False)) - _np(
        sma_ind(median, length=slow, use_talib=False)
    )
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_ao_swap_symmetry(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Slow < fast is swapped, results identical."""
    high, low = _hl_arrays(prices_random_walk)
    a = ao_numpy(high, low, fast=5, slow=34, use_talib=False)
    b = ao_numpy(high, low, fast=34, slow=5, use_talib=False)
    assert_allclose(a, b, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_ao_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN until the slow SMA warms up (slow - 1)."""
    high, low = _hl_arrays(prices_random_walk)
    fast, slow = 5, 34
    result = ao_numpy(high, low, fast=fast, slow=slow, use_talib=False)
    assert np.isnan(result[: slow - 1]).all()
    assert np.isfinite(result[slow - 1 :]).all()


@pytest.mark.momentum
def test_ao_non_contiguous_and_readonly(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views and read-only polars buffers must not change results.

    Regression: the old `for arr in (...): arr = ascontiguousarray(arr)`
    loop rebound the loop variable (no-op).
    """
    high, low = _hl_arrays(prices_random_walk)
    expected = ao_numpy(high, low, fast=5, slow=10, use_talib=False)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    assert not mk_nc(high).flags.c_contiguous
    assert_allclose(
        ao_numpy(
            mk_nc(high),
            mk_nc(low),
            fast=5,
            slow=10,
            use_talib=False,
        ),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        ao_ind(
            pl.Series(high),
            pl.Series(low),
            fast=5,
            slow=10,
            use_talib=False,
        ),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_ao_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Fast < 1 and slow < 1 are rejected."""
    high, low = _hl_arrays(prices_random_walk)
    with pytest.raises(ValueError, match="fast must be >= 1"):
        ao_numpy(high, low, fast=0, use_talib=False)
    with pytest.raises(ValueError, match="slow must be >= 1"):
        ao_numpy(high, low, fast=5, slow=0, use_talib=False)


@pytest.mark.momentum
def test_ao_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    high, low = _hl_arrays(prices_random_walk)
    slow = 10
    r0 = ao_numpy(high, low, fast=5, slow=slow, use_talib=False)
    r2 = ao_numpy(
        high,
        low,
        fast=5,
        slow=slow,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(
        r2[slow + 1 :],
        r0[slow - 1 : -2],
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_ao_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """ao_polars default column is AO_{fast}_{slow}."""
    result = ao_polars(df_ohlc, fast=5, slow=34, use_talib=False)
    assert "AO_5_34" in result.columns
    assert result["AO_5_34"].dtype == pl.Float64
    assert len(result) == len(df_ohlc)
    expected = ao_numpy(
        df_ohlc["high"].to_numpy(),
        df_ohlc["low"].to_numpy(),
        fast=5,
        slow=34,
        use_talib=False,
    )
    assert_allclose(result["AO_5_34"].to_numpy(), expected, equal_nan=True)
