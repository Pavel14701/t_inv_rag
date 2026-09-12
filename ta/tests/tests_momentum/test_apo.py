# -*- coding: utf-8 -*-
"""Unit tests for Absolute Price Oscillator (APO) module.

Covers:
- formula parity fast_ma - slow_ma (sma and ema modes)
- fast/slow swap symmetry
- warm-up NaNs follow the slow MA
- parameter validation (fast, slow)
- offset / fillna semantics
- apo_ind with pl.Series
- apo_polars DataFrame integration
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.ma import ma_mode
from ta.src.momentum.apo import apo_ind, apo_numpy, apo_polars
from ta.src.overlap.sma import sma_ind


def _np(arr: object) -> np.ndarray:
    """Narrow the ma/sma return to np.ndarray."""
    assert isinstance(arr, np.ndarray)
    return arr


@pytest.mark.momentum
def test_apo_matches_formula_sma(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Apo equals SMA(close, fast) - SMA(close, slow) exactly."""
    close = prices_random_walk
    fast, slow = 12, 26
    result = apo_numpy(
        close, fast=fast, slow=slow, mamode="sma", use_talib=False
    )
    expected = _np(sma_ind(close, length=fast, use_talib=False)) - _np(
        sma_ind(close, length=slow, use_talib=False)
    )
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_apo_ema_mode_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """mamode='ema' delegates to the same ma_mode backend."""
    close = prices_random_walk
    result = apo_numpy(close, fast=5, slow=10, mamode="ema", use_talib=False)
    expected = _np(
        ma_mode("ema", close, length=5, offset=0, fillna=None)
    ) - _np(ma_mode("ema", close, length=10, offset=0, fillna=None))
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_apo_swap_symmetry(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Slow < fast is swapped, results identical."""
    close = prices_random_walk
    a = apo_numpy(close, fast=12, slow=26, mamode="sma", use_talib=False)
    b = apo_numpy(close, fast=26, slow=12, mamode="sma", use_talib=False)
    assert_allclose(a, b, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_apo_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN until the slow MA warms up (slow - 1)."""
    close = prices_random_walk
    fast, slow = 12, 26
    result = apo_numpy(
        close, fast=fast, slow=slow, mamode="sma", use_talib=False
    )
    assert np.isnan(result[: slow - 1]).all()
    assert np.isfinite(result[slow - 1 :]).all()


@pytest.mark.momentum
def test_apo_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Fast < 1 and slow < 1 are rejected."""
    with pytest.raises(ValueError, match="fast must be >= 1"):
        apo_numpy(prices_random_walk, fast=0, use_talib=False)
    with pytest.raises(ValueError, match="slow must be >= 1"):
        apo_numpy(prices_random_walk, fast=5, slow=0, use_talib=False)


@pytest.mark.momentum
def test_apo_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    close = prices_random_walk
    slow = 10
    r0 = apo_numpy(close, fast=5, slow=slow, use_talib=False)
    r2 = apo_numpy(
        close,
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
def test_apo_ind_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """apo_ind accepts a read-only polars Series buffer."""
    close = prices_random_walk
    expected = apo_numpy(close, fast=5, slow=10, use_talib=False)
    result = apo_ind(pl.Series(close), fast=5, slow=10, use_talib=False)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_apo_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """apo_polars default column is APO_{fast}_{slow}."""
    result = apo_polars(df_ohlc, fast=12, slow=26, use_talib=False)
    assert "APO_12_26" in result.columns
    assert result["APO_12_26"].dtype == pl.Float64
    assert len(result) == len(df_ohlc)
    expected = apo_numpy(
        df_ohlc["close"].to_numpy(),
        fast=12,
        slow=26,
        use_talib=False,
    )
    assert_allclose(result["APO_12_26"].to_numpy(), expected, equal_nan=True)
