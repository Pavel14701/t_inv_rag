# -*- coding: utf-8 -*-
"""Unit tests for Schaff Trend Cycle (STC)."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...momentum.stc import (
    _ema_of_warmup,
    _stoch_of_series,
    stc_ind,
    stc_numpy,
    stc_polars,
)
from ...overlap.ema import ema_ind


def _stoch_reference(x: np.ndarray, window: int) -> np.ndarray:
    """Pure numpy rolling stochastic of a series."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = x[i - window + 1:i + 1]
        if np.isnan(w).any():
            continue
        denom = w.max() - w.min()
        if denom == 0.0:
            continue
        out[i] = 100.0 * (x[i] - w.min()) / denom
    return out


def _stc_reference(
    close: np.ndarray,
    tclen: int = 10,
    fast: int = 12,
    slow: int = 26,
    factor: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure numpy STC reference with the same warm-up seeding."""
    macd = (
        ema_ind(close, length=fast, use_talib=False, nan_policy='ignore')
        - ema_ind(close, length=slow, use_talib=False, nan_policy='ignore')
    )

    def seeded_ema(x: np.ndarray, length: int) -> np.ndarray:
        out = np.full(len(x), np.nan)
        first_valid = np.argmax(~np.isnan(x))
        if np.isnan(x[first_valid]):
            return out
        filled = x.copy()
        filled[:first_valid] = x[first_valid]
        res = ema_ind(filled, length=length, use_talib=False,
                      nan_policy='ignore')
        res[:first_valid + length - 1] = np.nan
        return res

    stoch = seeded_ema(_stoch_reference(macd, tclen), factor)
    stc = seeded_ema(_stoch_reference(stoch, tclen), factor)
    return stc, macd, stoch


@pytest.mark.momentum
def test_stc_matches_reference(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    expected = _stc_reference(close)
    result = stc_numpy(close)
    for res, exp in zip(result, expected):
        assert_allclose(res, exp, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_stc_warmup_and_bounds(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    stc, macd, stoch = stc_numpy(close)
    # MACD valid from slow - 1 = 25.
    assert np.isnan(macd[:25]).all()
    assert not np.isnan(macd[25:]).any()
    # Stage 1 valid from 25 + 9 + 2 = 36.
    assert np.isnan(stoch[:36]).all()
    assert not np.isnan(stoch[36:]).any()
    # STC valid from 36 + 9 + 2 = 47.
    assert np.isnan(stc[:47]).all()
    assert not np.isnan(stc[47:]).any()
    assert ((stc[47:] >= 0.0) & (stc[47:] <= 100.0)).all()


@pytest.mark.momentum
def test_stoch_of_series_flat_window_is_nan() -> None:
    x = np.full(20, 5.0)
    result = _stoch_of_series(x, 5)
    assert np.isnan(result[4:]).all()


@pytest.mark.momentum
def test_stoch_of_series_ends_match() -> None:
    x = np.linspace(1.0, 20.0, 20)
    result = _stoch_of_series(x, 5)
    # Rising series: every value is the window max -> 100.
    assert_allclose(result[4:], 100.0, rtol=1e-12)


@pytest.mark.momentum
def test_ema_of_warmup_masks_prefix() -> None:
    x = np.full(30, np.nan)
    x[10:] = np.arange(20.0)
    result = _ema_of_warmup(x, 4)
    assert np.isnan(result[:13]).all()
    assert not np.isnan(result[13:]).any()
    # Compare with direct EMA on the filled series.
    filled = x.copy()
    filled[:10] = x[10]
    expected = ema_ind(filled, length=4, use_talib=False,
                       nan_policy='ignore')
    expected[:13] = np.nan
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_ema_of_warmup_all_nan() -> None:
    result = _ema_of_warmup(np.full(10, np.nan), 3)
    assert np.isnan(result).all()


@pytest.mark.momentum
@pytest.mark.parametrize('kw', [
    {'tclen': 0}, {'fast': 0}, {'slow': -1}, {'factor': 0},
])
def test_stc_invalid_params(kw: dict) -> None:
    with pytest.raises(ValueError):
        stc_numpy(np.arange(60.0), **kw)


@pytest.mark.momentum
def test_stc_empty_input() -> None:
    result = stc_numpy(np.array([], dtype=np.float64))
    for arr in result:
        assert arr.size == 0


@pytest.mark.momentum
def test_stc_offset_fillna(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    base = stc_numpy(close)
    shifted = stc_numpy(close, offset=2, fillna=50.0)
    for res, exp in zip(shifted, base):
        # fillna replaces both shifted-in positions and warm-up NaNs.
        expected = np.where(np.isnan(exp), 50.0, exp)
        assert np.all(res[:2] == 50.0)
        assert_allclose(res[2:], expected[:-2], rtol=1e-12)


@pytest.mark.momentum
def test_stc_ind_numpy_and_series(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    expected = stc_numpy(close)
    from_arrays = stc_ind(close)
    from_series = stc_ind(pl.Series(close))
    for res, exp in zip(from_arrays, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    for res, exp in zip(from_series, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_stc_polars(df_random_walk: pl.DataFrame) -> None:
    close = df_random_walk['close'].to_numpy()
    stc, macd, stoch = stc_numpy(close)
    result = stc_polars(df_random_walk)
    for col, exp in (
        ('STC_10_12_26_3', stc),
        ('STCmacd_10_12_26_3', macd),
        ('STCstoch_10_12_26_3', stoch),
    ):
        assert col in result.columns
        assert_allclose(
            result[col].to_numpy(), exp, rtol=1e-12, equal_nan=True
        )


@pytest.mark.momentum
def test_stc_readonly_input(prices_random_walk) -> None:
    arr = prices_random_walk.copy()
    arr.setflags(write=False)
    expected = stc_numpy(prices_random_walk)
    for res, exp in zip(stc_numpy(arr), expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
