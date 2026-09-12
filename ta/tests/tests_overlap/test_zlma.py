# -*- coding: utf-8 -*-
"""Unit tests for ZLMA (Zero Lag Moving Average).
Uses conftest fixtures. Covers default mode, all registered mamodes,
offset/fillna, polars integration, and IEEE 754 compliance.
"""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.overlap.zlma import _MA_FUNCS, _call_ma, zlma_ind, zlma_polars


@pytest.mark.overlap
def test_zlma_ind_default_ema(prices_random_walk) -> None:
    """Default mamode='ema' produces a usable series."""
    result = zlma_ind(prices_random_walk, length=10, use_talib=False)
    assert len(result) == len(prices_random_walk)
    # lag = int(0.5*(10-1)) = 4; warmup from EMA covers 9 leading values
    assert np.isnan(result[:9]).all()
    assert np.isfinite(result[9:]).all()


@pytest.mark.overlap
@pytest.mark.parametrize("mamode", sorted(_MA_FUNCS.keys()))
def test_zlma_ind_all_modes(prices_random_walk, mamode) -> None:
    """Every registered mamode runs and yields a finite tail on clean data."""
    result = zlma_ind(
        prices_random_walk, length=10, mamode=mamode, use_talib=False
    )
    assert len(result) == len(prices_random_walk)
    assert np.isfinite(result[-1])


@pytest.mark.overlap
def test_zlma_ind_invalid_mamode() -> None:
    """Unknown mamode raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported type of MA"):
        zlma_ind(np.linspace(1.0, 50.0, 50), length=5, mamode="bogus")


@pytest.mark.overlap
def test_zlma_ind_offset_fillna(prices_random_walk) -> None:
    """offset/fillna are applied via _apply_offset_fillna."""
    close = prices_random_walk
    base = zlma_ind(
        close, length=10, mamode="wma", offset=0, fillna=None, use_talib=False
    )
    expected = _apply_offset_fillna(base, 1, 0.0)
    result = zlma_ind(
        close, length=10, mamode="wma", offset=1, fillna=0.0, use_talib=False
    )
    assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


@pytest.mark.overlap
def test_zlma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """zlma_polars adds the ZLMA column."""
    result = zlma_polars(
        df_random_walk, length=10, mamode="ema", use_talib=False
    )
    col = "ZL_EMA_10"
    assert col in result.columns
    assert len(result) == len(df_random_walk)
    expected = zlma_ind(
        df_random_walk["close"].to_numpy(),
        length=10,
        mamode="ema",
        use_talib=False,
    )
    assert_allclose(
        result[col].to_numpy(), expected, rtol=1e-9, equal_nan=True
    )


@pytest.mark.overlap
def test_zlma_call_ma_kwargs() -> None:
    """_call_ma forwards only kwargs the backend supports."""
    from ta.src.overlap.pwma import pwma_ind
    from ta.src.overlap.sma import sma_ind

    arr = np.linspace(1.0, 50.0, 50)
    got = _call_ma(sma_ind, arr, 5, use_talib=False)
    expected = sma_ind(arr, 5, use_talib=False, nan_policy="ignore")
    assert_allclose(got, expected, rtol=1e-12, equal_nan=True)
    got2 = _call_ma(pwma_ind, arr, 5, use_talib=False)
    expected2 = pwma_ind(arr, 5)
    assert_allclose(got2, expected2, rtol=1e-12, equal_nan=True)


# ---- IEEE 754 (using conftest fixtures) ----


def test_zlma_with_nan(prices_with_nan):
    """NaN in input propagates; output has no Inf escalation."""
    result = zlma_ind(prices_with_nan, length=5, mamode="wma", use_talib=False)
    assert not np.isinf(result).any()


def test_zlma_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN; no Inf in output."""
    result = zlma_ind(prices_with_inf, length=5, mamode="ema", use_talib=False)
    assert not np.isinf(result).any()


def test_zlma_all_nan(prices_all_nan):
    """All-NaN input stays all-NaN (or fillna if provided)."""
    result = zlma_ind(prices_all_nan, length=3, mamode="wma", use_talib=False)
    assert np.isnan(result).all()
    filled = zlma_ind(
        prices_all_nan, length=3, mamode="wma", fillna=0.0, use_talib=False
    )
    assert (filled == 0.0).all()


def test_zlma_empty(prices_empty):
    """Empty input yields empty output without crashing."""
    result = zlma_ind(prices_empty, length=5, mamode="fwma", use_talib=False)
    assert len(result) == 0


def test_zlma_extreme_values(prices_extreme):
    """Extreme values must not crash and must not produce Inf."""
    result = zlma_ind(prices_extreme, length=3, mamode="wma", use_talib=False)
    assert not np.isinf(result).any()


def test_zlma_polars_with_nan(df_random_walk):
    """Polars integration propagates NaN without Inf."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = zlma_polars(df_with_nan, length=5, mamode="wma", use_talib=False)
    vals = result["ZL_WMA_5"].to_numpy()
    assert not np.isinf(vals).any()
    assert np.isnan(vals).any()
