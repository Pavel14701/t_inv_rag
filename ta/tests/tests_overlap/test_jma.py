# -*- coding: utf-8 -*-
"""Unit tests for Jurik Moving Average (JMA) module.

Tests cover:
- _jma_numba_core against reference implementation
- length validation
- warmup NaN behaviour
- offset and fillna
- jma_ind with Polars Series
- jma_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.overlap.jma import (
    _jma_numba_core,
    jma_ind,
    jma_numba,
    jma_polars,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------


def _jma_reference(
    close: npt.NDArray[np.float64],
    length: int,
    phase: float,
) -> npt.NDArray[np.float64]:
    """Pure Python reference implementation of JMA."""
    n = len(close)
    jma = np.empty(n, dtype=np.float64)
    if n == 0:
        return jma
    volty = np.empty(n, dtype=np.float64)
    v_sum = np.empty(n, dtype=np.float64)
    sum_length = 10
    l_half = 0.5 * (length - 1)
    if phase < -100.0:
        pr = 0.5
    elif phase > 100.0:
        pr = 2.5
    else:
        pr = 1.5 + phase * 0.01
    length1 = max(np.log(np.sqrt(l_half)) / np.log(2.0) + 2.0, 0.0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(l_half)
    bet = length2 / (length2 + 1.0)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    limit = length1 ** (1.0 / pow1)
    jma[0] = close[0]
    volty[0] = 0.0
    v_sum[0] = 0.0
    ma1 = close[0]
    u_band = close[0]
    l_band = close[0]
    det0 = 0.0
    det1 = 0.0
    window_size = 66
    v_sum_window = np.zeros(window_size, dtype=np.float64)
    v_sum_idx = 0
    v_sum_total = 0.0
    window_filled = False
    for i in range(1, n):
        price = close[i]
        del1 = price - u_band
        del2 = price - l_band
        if abs(del1) != abs(del2):
            volty[i] = max(abs(del1), abs(del2))
        else:
            volty[i] = 0.0
        past_idx = i - sum_length
        if past_idx < 0:
            past_idx = 0
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[past_idx]) / sum_length
        if i < window_size:
            v_sum_total += v_sum[i]
            v_sum_window[i] = v_sum[i]
        else:
            oldest = v_sum_window[v_sum_idx]
            v_sum_total = v_sum_total - oldest + v_sum[i]
            v_sum_window[v_sum_idx] = v_sum[i]
            v_sum_idx = (v_sum_idx + 1) % window_size
            window_filled = True
        if window_filled:
            avg_volty = v_sum_total / window_size
        else:
            avg_volty = v_sum_total / (i + 1) if i > 0 else 0.0
        d_volty = 0.0 if avg_volty == 0.0 else volty[i] / avg_volty
        if d_volty < 1.0:
            r_volty = 1.0
        elif d_volty > limit:
            r_volty = limit
        else:
            r_volty = d_volty
        power = r_volty**pow1
        kv = bet ** np.sqrt(power)
        if del1 > 0.0:
            u_band = price
        else:
            u_band = price - kv * del1
        if del2 < 0.0:
            l_band = price
        else:
            l_band = price - kv * del2
        alpha = beta**power
        ma1 = (1.0 - alpha) * price + alpha * ma1
        det0 = (1.0 - beta) * (price - ma1) + beta * det0
        ma2 = ma1 + pr * det0
        det1 = ((ma2 - jma[i - 1]) * (1.0 - alpha) * (1.0 - alpha)) + (
            alpha * alpha * det1
        )
        jma[i] = jma[i - 1] + det1
    for i in range(length - 1):
        jma[i] = np.nan
    return jma


# -----------------------------------------------------------------------------
# Tests for _jma_numba_core
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_jma_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test _jma_numba_core against pure Python reference."""
    close = prices_random_walk
    result = _jma_numba_core(close, 7, 0.0)
    expected = _jma_reference(close, 7, 0.0)
    assert_allclose(result, expected, rtol=1e-8, equal_nan=True)


@pytest.mark.overlap
def test_jma_core_empty() -> None:
    """Empty input returns empty array (no IndexError)."""
    result = _jma_numba_core(np.array([]), 7, 0.0)
    assert result.size == 0


@pytest.mark.overlap
def test_jma_core_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """First length-1 values are NaN, the rest are finite."""
    length = 7
    result = _jma_numba_core(prices_random_walk, length, 0.0)
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


# -----------------------------------------------------------------------------
# Tests for jma_numba
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_jma_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test jma_numba against pure Python reference."""
    close = prices_random_walk
    result = jma_numba(close, length=7, phase=0.0)
    expected = _jma_reference(close, 7, 0.0)
    assert_allclose(result, expected, rtol=1e-8, equal_nan=True)


@pytest.mark.overlap
def test_jma_numba_phase_effects(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Phase clamping outside [-100, 100] does not crash."""
    close = prices_random_walk
    r1 = jma_numba(close, length=7, phase=-150.0)
    r2 = jma_numba(close, length=7, phase=150.0)
    e1 = _jma_reference(close, 7, -150.0)
    e2 = _jma_reference(close, 7, 150.0)
    assert_allclose(r1, e1, rtol=1e-8, equal_nan=True)
    assert_allclose(r2, e2, rtol=1e-8, equal_nan=True)


@pytest.mark.overlap
def test_jma_numba_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        jma_numba(close, length=0)


@pytest.mark.overlap
def test_jma_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using the real _apply_offset_fillna."""
    close = prices_random_walk
    length = 7
    offset = 2
    fillna = 0.0
    base = jma_numba(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = jma_numba(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-8, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for jma_ind (universal wrapper)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_jma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test jma_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    result = jma_ind(s, length=7)
    expected = _jma_reference(prices_random_walk, 7, 0.0)
    assert_allclose(result, expected, rtol=1e-8, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for jma_polars (DataFrame integration)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_jma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test jma_polars adds a column correctly."""
    result_df = jma_polars(df_random_walk, length=7, output_col="JMA")
    assert "JMA" in result_df.columns
    assert result_df["JMA"].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk["close"].to_numpy()
    expected = _jma_reference(close_arr, 7, 0.0)
    assert_allclose(
        result_df["JMA"].to_numpy(), expected, rtol=1e-8, equal_nan=True
    )


@pytest.mark.overlap
def test_jma_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    result_df = jma_polars(df, length=7)
    assert "JMA_7_0.0" in result_df.columns


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_jma_numba_nan_policy_raise() -> None:
    """Input with NaN and default nan_policy='raise' raises ValueError."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="NaN"):
        jma_numba(data, length=3)


@pytest.mark.overlap
def test_jma_numba_nan_policy_ffill() -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    result = jma_numba(data, length=3, nan_policy="ffill")
    assert np.isfinite(result[2:]).all()


@pytest.mark.overlap
def test_jma_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    with pytest.raises(ValueError, match="NaN"):
        jma_numba(prices_with_inf)
    result = jma_numba(prices_with_inf, length=5, nan_policy="ffill")
    assert np.isfinite(result[4:]).all()


@pytest.mark.overlap
def test_jma_numba_empty(prices_empty):
    """Empty input returns empty array."""
    result = jma_numba(prices_empty)
    assert result.size == 0


@pytest.mark.overlap
def test_jma_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = jma_numba(prices_all_nan, length=5, nan_policy="ignore")
    assert np.isnan(result[4:]).all()
    result_fill = jma_numba(
        prices_all_nan, length=5, fillna=0.0, nan_policy="ignore"
    )
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_jma_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    result = jma_numba(prices_extreme, length=5, nan_policy="ignore")
    assert result is not None
