# -*- coding: utf-8 -*-
"""Unit tests for Percentage Price Oscillator (PPO)."""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from ta.src.external import talib, talib_available
from ta.src.momentum.ppo import ppo_ind, ppo_numpy, ppo_polars
from ta.src.overlap.ema import ema_ind


@pytest.fixture
def prices(prices_random_walk: np.ndarray) -> np.ndarray:
    return prices_random_walk


def _ppo_reference(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    scalar: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure numpy PPO reference (EMA seeding identical to ema_ind)."""
    fast_ema = ema_ind(
        close, length=fast, use_talib=False, nan_policy="ignore"
    )
    slow_ema = ema_ind(
        close, length=slow, use_talib=False, nan_policy="ignore"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        line = scalar * (fast_ema - slow_ema) / slow_ema
    filled = line.copy()
    first_valid = np.argmax(~np.isnan(line))
    if not np.isnan(line[first_valid]):
        filled[:first_valid] = line[first_valid]
    signalma = ema_ind(
        filled, length=signal, use_talib=False, nan_policy="ignore"
    )
    signalma[: slow + signal - 2] = np.nan
    hist = line - signalma
    return line, signalma, hist


@pytest.mark.momentum
def test_ppo_matches_reference(prices) -> None:
    expected = _ppo_reference(prices)
    line, signalma, hist = ppo_numpy(prices, use_talib=False)
    for res, exp in zip((line, signalma, hist), expected, strict=False):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_ppo_warmup_nan(prices) -> None:
    line, signalma, hist = ppo_numpy(prices)
    assert np.isnan(line[:25]).all()
    assert not np.isnan(line[25:]).any()
    assert np.isnan(signalma[:33]).all()
    assert np.isnan(hist[:33]).all()


@pytest.mark.momentum
@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
def test_ppo_line_matches_talib(prices) -> None:
    expected = talib.PPO(prices, 12, 26, 0)
    line, _, _ = ppo_numpy(prices)
    assert_allclose(line, expected, rtol=1e-10, equal_nan=True)
    # Custom scalar bypasses TA-Lib; scaling is exact on the native path.
    line_50, _, _ = ppo_numpy(prices, scalar=50.0, use_talib=False)
    line_100, _, _ = ppo_numpy(prices, use_talib=False)
    assert_allclose(line_50, 0.5 * line_100, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_ppo_native_matches_talib_tail(prices) -> None:
    # TA-Lib seeds its internal PPO EMAs relative to the lookback
    # (SMA over the last `period` bars ending at slow-1), so the native
    # and TA-Lib series never converge bitwise. Assert they track each
    # other: strong tail correlation and identical value range.
    line_native, _, _ = ppo_numpy(prices, use_talib=False)
    line_talib, _, _ = ppo_numpy(prices, use_talib=True)
    corr = np.corrcoef(line_native[60:], line_talib[60:])[0, 1]
    assert corr > 0.8
    assert_allclose(
        [line_native[60:].min(), line_native[60:].max()],
        [line_talib[60:].min(), line_talib[60:].max()],
        rtol=0.5,
        atol=0.5,
    )


@pytest.mark.momentum
def test_ppo_hist_equals_line_minus_signal(prices) -> None:
    line, signalma, hist = ppo_numpy(prices)
    valid = ~np.isnan(signalma)
    assert_allclose(hist[valid], (line - signalma)[valid], rtol=1e-12)


@pytest.mark.momentum
def test_ppo_short_input_returns_nan_arrays() -> None:
    close = np.array([1.0, 2.0])
    line, signalma, hist = ppo_numpy(close)
    for arr in (line, signalma, hist):
        assert np.isnan(arr).all()
        assert arr.size == 2


@pytest.mark.momentum
def test_ppo_empty_input() -> None:
    line, signalma, hist = ppo_numpy(np.array([], dtype=np.float64))
    for arr in (line, signalma, hist):
        assert arr.size == 0


@pytest.mark.momentum
def test_ppo_inf_input_becomes_nan(prices) -> None:
    close = prices.copy()
    close[30] = np.inf
    line, _, _ = ppo_numpy(close, use_talib=False)
    # Inf -> NaN, and NaN poisons the EMA recursion from that bar on.
    assert np.isnan(line[:25]).all()  # EMA warm-up
    assert np.isfinite(line[25:30]).all()
    assert np.isnan(line[30:]).all()


@pytest.mark.momentum
def test_ppo_offset_fillna(prices) -> None:
    line, signalma, hist = ppo_numpy(prices, offset=2, fillna=0.0)
    base_line, base_signal, base_hist = ppo_numpy(prices)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    exp_line = np.where(np.isnan(base_line), 0.0, base_line)
    exp_signal = np.where(np.isnan(base_signal), 0.0, base_signal)
    exp_hist = np.where(np.isnan(base_hist), 0.0, base_hist)
    assert_array_equal(line[:2], np.zeros(2))
    assert_allclose(line[2:], exp_line[:-2], rtol=1e-12)
    assert_allclose(signalma[2:], exp_signal[:-2], rtol=1e-12)
    assert_allclose(hist[2:], exp_hist[:-2], rtol=1e-12)


@pytest.mark.momentum
@pytest.mark.parametrize(
    "fast, slow, signal", [(0, 26, 9), (12, 0, 9), (12, 26, 0)]
)
def test_ppo_invalid_params(fast: int, slow: int, signal: int) -> None:
    with pytest.raises(ValueError):
        ppo_numpy(np.arange(50.0), fast=fast, slow=slow, signal=signal)


@pytest.mark.momentum
def test_ppo_ind_numpy_and_series(prices) -> None:
    expected = ppo_numpy(prices)
    from_arrays = ppo_ind(prices)
    from_series = ppo_ind(pl.Series(prices))
    for res, exp in zip(from_series, expected, strict=False):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    for res, exp in zip(from_arrays, expected, strict=False):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_ppo_polars(df_random_walk: pl.DataFrame) -> None:
    close = df_random_walk["close"].to_numpy()
    line, signalma, hist = ppo_numpy(close)
    result = ppo_polars(df_random_walk)
    for col, exp in (
        ("PPO_12_26_9", line),
        ("PPOs_12_26_9", signalma),
        ("PPOh_12_26_9", hist),
    ):
        assert col in result.columns
        assert_allclose(
            result[col].to_numpy(), exp, rtol=1e-12, equal_nan=True
        )


@pytest.mark.momentum
def test_ppo_readonly_input(prices) -> None:
    arr = prices.copy()
    arr.setflags(write=False)
    expected = ppo_numpy(prices)
    result = ppo_numpy(arr)
    for res, exp in zip(result, expected, strict=False):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
