# -*- coding: utf-8 -*-
"""Unit tests for Weighted Moving Average (WMA) module.

Tests cover:
- Weight generation (_get_wma_weights) with cache, normalisation, asc/desc
- wma_numba against reference implementation
- offset and fillna
- wma_talib (if TA-Lib available) against reference
- wma_ind with Polars Series
- wma_polars DataFrame integration
- trim functionality
- IEEE 754 compliance (NaN, Inf, empty, extreme)
- Backend selection (use_talib)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_almost_equal

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.wma import (
    _get_wma_weights,
    wma_ind,
    wma_numba,
    wma_polars,
    wma_talib,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _wma_reference(
    close: npt.NDArray[np.float64],
    length: int,
    asc: bool = True,
) -> npt.NDArray[np.float64]:
    """Pure Python reference WMA."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    w = np.arange(1, length + 1, dtype=np.float64)
    if not asc:
        w = w[::-1]
    w /= w.sum()
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += close[i - j] * w[length - 1 - j]
        out[i] = acc
    return out


# -----------------------------------------------------------------------------
# Tests for weight generation
# -----------------------------------------------------------------------------
@pytest.mark.overlap
@pytest.mark.parametrize("asc", [True, False], ids=["asc", "desc"])
def test_wma_weights_sum_to_one(asc: bool) -> None:
    """Test that generated weights sum to 1 for both directions."""
    length = 10
    weights = _get_wma_weights(length, asc)
    assert_almost_equal(weights.sum(), 1.0, decimal=6)


@pytest.mark.overlap
@pytest.mark.parametrize("asc", [True, False], ids=["asc", "desc"])
def test_wma_weights_direction(asc: bool) -> None:
    """Test that asc=True gives higher weight to recent values."""
    length = 10
    weights = _get_wma_weights(length, asc)
    weights_other = _get_wma_weights(length, not asc)
    # Direction: recent weight is larger for asc, smaller for desc.
    assert weights[-1] > weights[0] if asc else weights[-1] < weights[0]
    # Same length: asc and desc are exact reverses.
    assert np.array_equal(weights, weights_other[::-1])


@pytest.mark.overlap
def test_wma_weights_cache() -> None:
    """Test that weights are cached (lru_cache returns same object)."""
    length = 10
    asc = True
    w1 = _get_wma_weights(length, asc)
    w2 = _get_wma_weights(length, asc)
    assert w1 is w2


@pytest.mark.overlap
def test_wma_weights_readonly() -> None:
    """Cached weights must be read-only to protect the lru_cache."""
    weights = _get_wma_weights(10, True)
    assert not weights.flags.writeable
    with pytest.raises(ValueError):
        weights[0] = 1.0


@pytest.mark.overlap
def test_wma_numba_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        wma_numba(close, length=0)


@pytest.mark.overlap
def test_wma_numba_invalid_nan_policy() -> None:
    """Unknown nan_policy raises ValueError."""
    close = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match="nan_policy"):
        wma_numba(close, length=2, nan_policy="invalid")


# -----------------------------------------------------------------------------
# Tests for wma_numba
# -----------------------------------------------------------------------------
@pytest.mark.overlap
@pytest.mark.parametrize(
    "length,asc",
    [(5, True), (5, False), (20, True), (20, False)],
    ids=["len5-asc", "len5-desc", "len20-asc", "len20-desc"],
)
def test_wma_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
    length: int,
    asc: bool,
) -> None:
    """Test wma_numba against pure Python reference for both directions."""
    close = prices_random_walk
    result = wma_numba(close, length=length, asc=asc)
    expected = _wma_reference(close, length, asc)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_wma_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using _apply_offset_fillna."""
    close = prices_random_walk
    length = 10
    asc = True
    offset = 3
    fillna = 0.0
    base = wma_numba(close, length=length, asc=asc, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = wma_numba(
        close, length=length, asc=asc, offset=offset, fillna=fillna
    )
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_wma_numba_short_window() -> None:
    """Window longer than data raises ValueError."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    length = 5
    with pytest.raises(ValueError, match="Input series too short"):
        wma_numba(close, length=length)


@pytest.mark.overlap
def test_wma_numba_trim(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Trim removes first (length-1) values."""
    close = prices_random_walk
    length = 10
    asc = True
    result = wma_numba(close, length=length, asc=asc, trim=True)
    expected_len = len(close) - (length - 1)
    assert len(result) == expected_len
    full = wma_numba(close, length=length, asc=asc, trim=False)
    assert_allclose(result, full[length - 1 :], rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_wma_numba_trim_with_offset(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Trim with offset applied correctly."""
    close = prices_random_walk
    length = 10
    asc = True
    offset = 3
    fillna = 0.0
    base = _wma_reference(close, length, asc)
    base_trimmed = base[length - 1 :]
    expected = _apply_offset_fillna(base_trimmed, offset, fillna)
    result = wma_numba(
        close, length=length, asc=asc, offset=offset, fillna=fillna, trim=True
    )
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for wma_talib (if available)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_wma_talib_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """wma_talib matches reference (asc=True only)."""
    close = prices_random_walk
    length = 10
    asc = True
    result = wma_talib(close, length=length)
    expected = _wma_reference(close, length, asc)
    # TA-Lib may have slightly different initialisation, allow larger tolerance
    mask = np.isfinite(result) & np.isfinite(expected)
    assert_allclose(result[mask], expected[mask], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_wma_talib_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """wma_talib offset and fillna work correctly."""
    close = prices_random_walk
    length = 10
    offset = 3
    fillna = 0.0
    base = wma_talib(close, length=length, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = wma_talib(close, length=length, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_wma_talib_trim(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """wma_talib trim removes first (length-1) values."""
    close = prices_random_walk
    length = 10
    result = wma_talib(close, length=length, trim=True)
    expected_len = len(close) - (length - 1)
    assert len(result) == expected_len
    full = wma_talib(close, length=length, trim=False)
    assert_allclose(result, full[length - 1 :], rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for wma_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_wma_ind_uses_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """wma_ind with use_talib=False uses Numba."""
    close = prices_random_walk
    length = 10
    asc = True
    result = wma_ind(close, length=length, asc=asc, use_talib=False)
    expected = _wma_reference(close, length, asc)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_wma_ind_uses_talib(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """wma_ind with use_talib=True and asc=True uses TA-Lib."""
    close = prices_random_walk
    length = 10
    asc = True
    result = wma_ind(close, length=length, asc=asc, use_talib=True)
    expected = wma_talib(close, length=length)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_wma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """wma_ind accepts Polars Series."""
    s = pl.Series(prices_random_walk)
    length = 10
    asc = True
    result = wma_ind(s, length=length, asc=asc, use_talib=False)
    expected = _wma_reference(prices_random_walk, length, asc)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_wma_ind_asc_false_uses_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """asc=False forces Numba (TA-Lib doesn't support it)."""
    close = prices_random_walk
    length = 10
    asc = False
    result = wma_ind(close, length=length, asc=asc, use_talib=True)
    expected = _wma_reference(close, length, asc)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for wma_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_wma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """wma_polars adds correct column."""
    length = 10
    asc = True
    result_df = wma_polars(
        df_random_walk,
        close_col="close",
        length=length,
        asc=asc,
        use_talib=False,
        output_col="WMA",
    )
    assert "WMA" in result_df.columns
    assert result_df["WMA"].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk["close"].to_numpy()
    expected = _wma_reference(close_arr, length, asc)
    assert_allclose(
        result_df["WMA"].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_wma_polars_default_output_col() -> None:
    """Default output column name is f'WMA_{length}'."""
    df = pl.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    length = 3
    result_df = wma_polars(
        df, close_col="close", length=length, use_talib=False
    )
    expected_col = f"WMA_{length}"
    assert expected_col in result_df.columns


@pytest.mark.overlap
def test_wma_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """wma_polars applies offset and fillna."""
    length = 10
    asc = True
    offset = 2
    fillna = 0.0
    result_df = wma_polars(
        df_random_walk,
        close_col="close",
        length=length,
        asc=asc,
        offset=offset,
        fillna=fillna,
        use_talib=False,
        output_col="WMA",
    )
    close_arr = df_random_walk["close"].to_numpy()
    expected = wma_numba(
        close_arr, length=length, asc=asc, offset=offset, fillna=fillna
    )
    assert_allclose(
        result_df["WMA"].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
def test_wma_numba_with_nan(prices_with_nan):
    """NaN in input propagates through WMA."""
    length = 5
    asc = True
    result = wma_numba(
        prices_with_nan, length=length, asc=asc, nan_policy="ignore"
    )
    # NaN at index 5. With length=5, NaN will appear from index 5..9.
    assert np.isnan(result[5:10]).all()
    # After index 10, no NaN should remain (window no longer includes NaN)
    assert np.isfinite(result[10:]).all()


def test_wma_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, behaves like NaN."""
    length = 5
    asc = True
    result = wma_numba(
        prices_with_inf, length=length, asc=asc, nan_policy="ignore"
    )
    assert np.isnan(result[5:10]).all()
    assert np.isfinite(result[10:]).all()


def test_wma_numba_empty(prices_empty):
    """Empty input raises ValueError (series too short)."""
    with pytest.raises(ValueError, match="Input series too short"):
        wma_numba(prices_empty, length=5)


def test_wma_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = wma_numba(prices_all_nan, length=5, nan_policy="ignore")
    assert np.isnan(result).all()
    result_fill = wma_numba(
        prices_all_nan, length=5, fillna=0.0, nan_policy="ignore"
    )
    assert (result_fill == 0.0).all()


def test_wma_numba_extreme_values(prices_extreme):
    """Extreme values must not crash."""
    length = 5
    result = wma_numba(prices_extreme, length=length, nan_policy="ignore")
    assert result is not None


def test_wma_polars_with_nan(df_random_walk):
    """Polars integration should propagate NaN correctly (using Numba)."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([pl.Series("close", close_arr)])
    result_df = wma_polars(
        df_with_nan,
        close_col="close",
        length=5,
        output_col="WMA",
        use_talib=False,  # force Numba; TA-Lib can't handle NaN correctly
        nan_policy="ignore",
    )
    assert "WMA" in result_df.columns
    wma_vals = result_df["WMA"].to_numpy()
    # NaN at index 5 affects windows 5..9
    assert np.isnan(wma_vals[5:10]).all()
    # from index 10 onwards, values should be finite
    assert np.isfinite(wma_vals[10:]).all()
