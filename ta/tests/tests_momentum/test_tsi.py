# -*- coding: utf-8 -*-
"""Unit tests for True Strength Index (TSI)."""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from ta.src.momentum.tsi import (
    _double_ema,
    tsi_ind,
    tsi_numpy,
    tsi_polars,
)
from ta.src.overlap.ema import ema_ind


def _tsi_reference(
    close: np.ndarray,
    long: int = 25,
    short: int = 13,
    signal: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure numpy TSI reference with the same warm-up seeding."""
    n = len(close)
    mom = np.full(n, np.nan)
    mom[1:] = np.diff(close)

    def ema_seed(x: np.ndarray, length: int) -> np.ndarray:
        out = np.full(n, np.nan)
        first_valid = np.argmax(~np.isnan(x))
        if np.isnan(x[first_valid]):
            return out
        filled = x.copy()
        filled[:first_valid] = x[first_valid]
        res = ema_ind(
            filled, length=length, use_talib=False, nan_policy="ignore"
        )
        res[: first_valid + length - 1] = np.nan
        return res

    def double(x: np.ndarray) -> np.ndarray:
        return ema_seed(ema_seed(x, short), long)

    num = double(mom)
    den = double(np.abs(mom))
    with np.errstate(divide="ignore", invalid="ignore"):
        tsi = 100.0 * num / den
    tsi = np.where(den == 0.0, np.nan, tsi)
    signalma = ema_seed(tsi, signal)
    return tsi, signalma


@pytest.mark.momentum
def test_tsi_matches_reference(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    expected_tsi, expected_signal = _tsi_reference(close)
    tsi, signalma = tsi_numpy(close)
    assert_allclose(tsi, expected_tsi, rtol=1e-10, equal_nan=True)
    assert_allclose(signalma, expected_signal, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_tsi_warmup_nan(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    tsi, signalma = tsi_numpy(close)
    # mom valid from 1; EMA(short=13) -> 13; EMA(long=25) -> 37.
    assert np.isnan(tsi[:37]).all()
    assert not np.isnan(tsi[37:]).any()
    # Signal EMA(13) of tsi valid from 37 + 13 - 1 = 49.
    assert np.isnan(signalma[:49]).all()
    assert not np.isnan(signalma[49:]).any()


@pytest.mark.momentum
def test_tsi_double_ema_helper(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    mom = np.diff(close, prepend=np.nan)
    result = _double_ema(mom, 5, 10)
    # Second stage valid from first_valid(=1) + 5 - 1 + 10 - 1 = 14.
    assert np.isnan(result[:14]).all()
    assert not np.isnan(result[14:]).any()


@pytest.mark.momentum
def test_tsi_flat_market_zero_denominator() -> None:
    close = np.full(60, 100.0)
    tsi, _ = tsi_numpy(close)
    # den == 0 -> NaN (undefined), never +/-Inf.
    assert np.isnan(tsi).all()


@pytest.mark.momentum
def test_tsi_bounded_and_signed(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    tsi, _ = tsi_numpy(close)
    valid = tsi[37:]
    # Numerically bounded by +-100 (denominator >= |numerator|).
    assert (np.abs(valid) <= 100.0 + 1e-9).all()
    # Strong final rally -> positive TSI.
    close2 = close.copy()
    close2[-8:] = close2[-9] + np.cumsum(np.full(8, 3.0))
    tsi2, _ = tsi_numpy(close2)
    assert tsi2[-1] > 0.0


@pytest.mark.momentum
def test_tsi_signal_is_ema_of_tsi(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    tsi, signalma = tsi_numpy(close)
    first_valid = np.argmax(~np.isnan(tsi))
    filled = tsi.copy()
    filled[:first_valid] = tsi[first_valid]
    expected = ema_ind(filled, length=13, use_talib=False, nan_policy="ignore")
    expected[: first_valid + 12] = np.nan
    assert_allclose(signalma, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
@pytest.mark.parametrize(
    "kw",
    [
        {"long": 0},
        {"short": -1},
        {"signal": 0},
    ],
)
def test_tsi_invalid_params(kw: dict) -> None:
    with pytest.raises(ValueError):
        tsi_numpy(np.arange(50.0), **kw)


@pytest.mark.momentum
def test_tsi_empty_input() -> None:
    tsi, signalma = tsi_numpy(np.array([], dtype=np.float64))
    assert tsi.size == 0
    assert signalma.size == 0


@pytest.mark.momentum
def test_tsi_all_nan_input() -> None:
    tsi, signalma = tsi_numpy(np.full(60, np.nan))
    assert np.isnan(tsi).all()
    assert np.isnan(signalma).all()


@pytest.mark.momentum
def test_tsi_offset_fillna(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    base_tsi, base_signal = tsi_numpy(close)
    tsi, signalma = tsi_numpy(close, offset=2, fillna=0.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    exp_tsi = np.where(np.isnan(base_tsi), 0.0, base_tsi)
    exp_signal = np.where(np.isnan(base_signal), 0.0, base_signal)
    assert_array_equal(tsi[:2], np.zeros(2))
    assert_allclose(tsi[2:], exp_tsi[:-2], rtol=1e-12)
    assert_allclose(signalma[2:], exp_signal[:-2], rtol=1e-12)


@pytest.mark.momentum
def test_tsi_ind_numpy_and_series(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    expected = tsi_numpy(close)
    from_arrays = tsi_ind(close)
    from_series = tsi_ind(pl.Series(close))
    for res, exp in zip(from_arrays, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    for res, exp in zip(from_series, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_tsi_polars(df_random_walk: pl.DataFrame) -> None:
    close = df_random_walk["close"].to_numpy()
    tsi, signalma = tsi_numpy(close)
    result = tsi_polars(df_random_walk)
    assert "TSI_25_13" in result.columns
    assert "TSIs_25_13" in result.columns
    assert_allclose(
        result["TSI_25_13"].to_numpy(), tsi, rtol=1e-12, equal_nan=True
    )
    assert_allclose(
        result["TSIs_25_13"].to_numpy(),
        signalma,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_tsi_readonly_input(prices_random_walk) -> None:
    arr = prices_random_walk.copy()
    arr.setflags(write=False)
    expected = tsi_numpy(prices_random_walk)
    for res, exp in zip(tsi_numpy(arr), expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
