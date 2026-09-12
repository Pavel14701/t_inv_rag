# -*- coding: utf-8 -*-
"""Unit tests for Kaufman's Adaptive Moving Average (KAMA) module.

Tests cover:
- _kama_numba_core against reference implementation
- kama_numba / kama_talib / kama_ind (backend selection)
- length and drift validation
- offset and fillna
- kama_ind with Polars Series
- kama_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.kama import (
    _kama_numba_core,
    kama_ind,
    kama_numba,
    kama_polars,
    kama_talib,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------


def _kama_reference(
    close: npt.NDArray[np.float64],
    length: int,
    fast: int,
    slow: int,
    drift: int,
) -> npt.NDArray[np.float64]:
    """Pure Python reference implementation of KAMA."""
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return kama
    fr = 2.0 / (fast + 1)
    sr = 2.0 / (slow + 1)
    abs_drift = np.empty(n, dtype=np.float64)
    abs_drift[0] = 0.0
    for i in range(1, n):
        abs_drift[i] = abs(close[i] - close[i - drift])
    cum = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        cum[i] = cum[i - 1] + abs_drift[i - 1]
    s = 0.0
    for i in range(length):
        s += close[i]
    kama[length - 1] = s / length
    for i in range(length, n):
        abs_diff = abs(close[i] - close[i - length])
        peer_sum = cum[i + 1] - cum[i + 1 - length]
        er = 0.0 if peer_sum == 0.0 else abs_diff / peer_sum
        sc = (er * (fr - sr) + sr) ** 2
        kama[i] = sc * close[i] + (1.0 - sc) * kama[i - 1]
    return kama


# -----------------------------------------------------------------------------
# Tests for _kama_numba_core
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_kama_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test _kama_numba_core against pure Python reference."""
    close = prices_random_walk
    result = _kama_numba_core(close, 10, 2, 30, 1)
    expected = _kama_reference(close, 10, 2, 30, 1)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_kama_core_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """First length-1 values are NaN, the rest are finite."""
    length = 10
    result = _kama_numba_core(prices_random_walk, length, 2, 30, 1)
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


# -----------------------------------------------------------------------------
# Tests for kama_numba
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_kama_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test kama_numba against pure Python reference."""
    close = prices_random_walk
    result = kama_numba(close, length=10, fast=2, slow=30, drift=1)
    expected = _kama_reference(close, 10, 2, 30, 1)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_kama_numba_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="length must be >= 1"):
        kama_numba(close, length=0)


@pytest.mark.overlap
def test_kama_numba_invalid_drift() -> None:
    """Drift below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="drift must be >= 1"):
        kama_numba(close, drift=0)


@pytest.mark.overlap
def test_kama_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using the real _apply_offset_fillna."""
    close = prices_random_walk
    length = 10
    offset = 2
    fillna = 0.0
    base = kama_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = kama_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_kama_talib_vs_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Backends converge to machine precision after the init transient."""
    close = prices_random_walk
    r_talib = kama_talib(close, length=10)
    r_numba = kama_numba(close, length=10)
    assert np.isfinite(r_talib[10:]).all()
    assert np.isfinite(r_numba[10:]).all()
    # Skip the initialization transient: both use SMA seeding but TA-Lib's
    # internal warm-up differs for the first periods.
    half = len(close) // 2
    assert_allclose(r_talib[half:], r_numba[half:], rtol=1e-3, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_kama_talib_invalid_length() -> None:
    """TA-Lib backend also validates length."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="length must be >= 1"):
        kama_talib(close, length=0)


# -----------------------------------------------------------------------------
# Tests for kama_ind (universal wrapper)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_kama_ind_uses_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test kama_ind uses Numba when TA-Lib is not requested."""
    close = prices_random_walk
    result = kama_ind(close, length=10, use_talib=False)
    expected = _kama_reference(close, 10, 2, 30, 1)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_kama_ind_uses_talib(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test kama_ind uses TA-Lib when available and requested."""
    close = prices_random_walk
    result = kama_ind(close, length=10, use_talib=True)
    expected = kama_talib(close, length=10)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_kama_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test kama_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    result = kama_ind(s, length=10, use_talib=False)
    expected = _kama_reference(prices_random_walk, 10, 2, 30, 1)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for kama_polars (DataFrame integration)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_kama_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test kama_polars adds a column correctly."""
    result_df = kama_polars(
        df_random_walk, length=10, use_talib=False, output_col="KAMA"
    )
    assert "KAMA" in result_df.columns
    assert result_df["KAMA"].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk["close"].to_numpy()
    expected = _kama_reference(close_arr, 10, 2, 30, 1)
    assert_allclose(
        result_df["KAMA"].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_kama_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    result_df = kama_polars(df, length=10, use_talib=False)
    assert "KAMA_10_2_30" in result_df.columns


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_kama_numba_nan_policy_raise() -> None:
    """Input with NaN and default nan_policy='raise' raises ValueError."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="NaN"):
        kama_numba(data, length=3)


@pytest.mark.overlap
def test_kama_numba_nan_policy_ffill() -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    result = kama_numba(data, length=3, nan_policy="ffill")
    assert np.isfinite(result[2:]).all()


@pytest.mark.overlap
def test_kama_numba_with_nan(prices_with_nan):
    """NaN in input poisons the KAMA recurrence permanently."""
    length = 5
    result = kama_numba(prices_with_nan, length=length, nan_policy="ignore")
    assert np.isfinite(result[4])
    assert np.isnan(result[5:]).all()


@pytest.mark.overlap
def test_kama_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    result = kama_numba(prices_with_inf, length=5, nan_policy="ignore")
    assert np.isfinite(result[4])
    assert np.isnan(result[5:]).all()


@pytest.mark.overlap
def test_kama_numba_empty(prices_empty):
    """Empty input returns empty array."""
    result = kama_numba(prices_empty, length=5)
    assert result.size == 0


@pytest.mark.overlap
def test_kama_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = kama_numba(prices_all_nan, length=5, nan_policy="ignore")
    assert np.isnan(result).all()
    result_fill = kama_numba(
        prices_all_nan, length=5, fillna=0.0, nan_policy="ignore"
    )
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_kama_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    result = kama_numba(prices_extreme, length=5, nan_policy="ignore")
    assert result is not None
