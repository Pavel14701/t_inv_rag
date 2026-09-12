# -*- coding: utf-8 -*-
"""Unit tests for Connors RSI (CRSI) module.

Covers:
- formula parity (RSI(price) + RSI(streak) + percent rank) / 3
- read-only polars buffers (regression: numba needs writable arrays)
- warm-up NaNs driven by rank_length
- parameter validation
- NaN input raises without nan_policy
- offset / fillna semantics
- crsi_polars DataFrame integration
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.momentum.crsi import crsi_ind, crsi_numpy, crsi_polars
from ta.src.momentum.rsi import rsi_ind


def _percent_rank(close: npt.NDArray[np.float64], length: int) -> np.ndarray:
    """Rolling percent rank reference (strictly-less counting)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = close[i - length + 1 : i + 1]
        out[i] = (window < close[i]).sum() / (length - 1) * 100.0
    return out


@pytest.mark.momentum
def test_crsi_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """CRSI equals the average of its three components."""
    close = prices_random_walk
    rsi_length, streak_length, rank_length = 3, 2, 50
    result = crsi_numpy(
        close, rsi_length, streak_length, rank_length, use_talib=False
    )
    rsi_price = np.asarray(rsi_ind(close, length=rsi_length, use_talib=False))
    # Streak reference
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    rsi_streak = np.asarray(
        rsi_ind(streak, length=streak_length, use_talib=False)
    )
    rank = _percent_rank(close, rank_length)
    expected = (rsi_price + rsi_streak + rank) / 3.0
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_crsi_readonly_polars_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Regression: pl.Series zero-copy buffers are read-only.

    Numba ``float64[:]`` signatures reject read-only arrays with a
    TypeError; the module must copy internally.
    """
    expected = crsi_numpy(prices_random_walk, use_talib=False)
    result = crsi_ind(pl.Series(prices_random_walk), use_talib=False)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_crsi_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN until the percent rank component warms up (rank_length - 1)."""
    result = crsi_numpy(
        prices_random_walk,
        rsi_length=3,
        streak_length=2,
        rank_length=100,
        use_talib=False,
    )
    assert np.isnan(result[:99]).all()
    assert np.isfinite(result[99:]).all()


@pytest.mark.momentum
def test_crsi_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """rsi_length < 1, streak_length < 1, rank_length < 2 are rejected."""
    with pytest.raises(ValueError, match="rsi_length must be >= 1"):
        crsi_numpy(prices_random_walk, rsi_length=0, use_talib=False)
    with pytest.raises(ValueError, match="streak_length must be >= 1"):
        crsi_numpy(prices_random_walk, streak_length=0, use_talib=False)
    with pytest.raises(ValueError, match="rank_length must be >= 2"):
        crsi_numpy(prices_random_walk, rank_length=1, use_talib=False)


@pytest.mark.momentum
def test_crsi_nan_policy(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN input raises by default; ffill produces finite results."""
    close = prices_random_walk.copy()
    close[10] = np.nan
    with pytest.raises(ValueError, match="Input contains NaN"):
        crsi_numpy(close, use_talib=False, nan_policy="raise")
    result = crsi_numpy(close, use_talib=False, nan_policy="ffill")
    assert np.isfinite(result[99:]).all()


@pytest.mark.momentum
def test_crsi_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    rank_length = 100
    r0 = crsi_numpy(
        prices_random_walk, rank_length=rank_length, use_talib=False
    )
    r2 = crsi_numpy(
        prices_random_walk,
        rank_length=rank_length,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(
        r2[rank_length + 1 :],
        r0[rank_length - 1 : -2],
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_crsi_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """crsi_polars default column is CRSI_{rsi}_{streak}_{rank}."""
    result = crsi_polars(df_ohlc, use_talib=False)
    assert "CRSI_3_2_100" in result.columns
    assert result["CRSI_3_2_100"].dtype == pl.Float64
    assert len(result) == len(df_ohlc)
    expected = crsi_numpy(
        df_ohlc["close"].to_numpy(),
        use_talib=False,
    )
    assert_allclose(
        result["CRSI_3_2_100"].to_numpy(), expected, equal_nan=True
    )
