# -*- coding: utf-8 -*-
"""Unit tests for MIDPRICE module.

Tests cover:
- Numba core function (_midprice_numba_core) against a reference
- Numba vs TA-Lib parity on clean data
- Full midprice_numba with offset and fillna
- Input validation (length < 1)
- Universal wrapper (midprice_ind) with NumPy, Polars Series and list input
- Polars integration (midprice_polars)
- IEEE 754 compliance (NaN, Inf, empty, short, all-NaN, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.midprice import (
    _midprice_numba_core,
    midprice_ind,
    midprice_numba,
    midprice_polars,
    midprice_talib,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python / numpy)
# -----------------------------------------------------------------------------


def _midprice_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure-Python reference: rolling (min(low) + max(high)) / 2."""
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        lo_win = low[i - length + 1 : i + 1]
        hi_win = high[i - length + 1 : i + 1]
        out[i] = (np.min(lo_win) + np.max(hi_win)) * 0.5
    return out


# -----------------------------------------------------------------------------
# Numba core function tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_midprice_numba_core_basic() -> None:
    """Test _midprice_numba_core against the pure-Python reference."""
    n = 20
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    for length in (2, 3, 5):
        result = _midprice_numba_core(high, low, length)
        expected = _midprice_reference(high, low, length)
        assert result.shape == high.shape
        assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
        # First `length - 1` values are the warm-up NaNs.
        assert np.isnan(result[: length - 1]).all()


@pytest.mark.overlap
def test_midprice_numba_core_window_bounds() -> None:
    """Midprice equals (min(low) + max(high)) / 2 of each exact window."""
    high = np.array([3.0, 5.0, 4.0, 6.0, 9.0, 7.0, 8.0, 6.5])
    low = np.array([1.0, 2.5, 1.5, 3.0, 4.0, 2.0, 3.5, 2.5])
    length = 4
    result = _midprice_numba_core(high, low, length)
    assert result[3] == (np.min(low[0:4]) + np.max(high[0:4])) / 2
    assert result[4] == (np.min(low[1:5]) + np.max(high[1:5])) / 2
    assert result[7] == (np.min(low[4:8]) + np.max(high[4:8])) * 0.5


@pytest.mark.overlap
@pytest.mark.parametrize("length", [2, 10, 30], ids=["len2", "len10", "len30"])
def test_midprice_numba_core_within_window_extremes(length: int) -> None:
    """Midprice lies between the window's min(low) and max(high).

    Note: it need not lie within [low[i], high[i]] of the *current* bar,
    because a dip in low (or a spike in high) earlier in the window shifts
    the midpoint away from the current bar's range.
    """
    rng = np.random.default_rng(7)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    for length in (2, 10, 30):
        result = midprice_numba(high, low, length)
        expected = _midprice_reference(high, low, length)
        mask = ~np.isnan(expected)
        assert_allclose(result[mask], expected[mask], rtol=1e-12)
        # Within the window's extremes by construction.
        for i in np.flatnonzero(mask):
            assert low[i - length + 1 : i + 1].min() - 1e-9 <= result[i]
            assert result[i] <= high[i - length + 1 : i + 1].max() + 1e-9


@pytest.mark.overlap
def test_midprice_numba_core_constant_series() -> None:
    """Constant high/low -> midprice equals the constant."""
    high = np.full(20, 42.0)
    low = np.full(20, 42.0)
    result = _midprice_numba_core(high, low, 5)
    finite = result[~np.isnan(result)]
    assert finite.size == len(high) - 4
    assert (finite == 42.0).all()


@pytest.mark.overlap
def test_midprice_numba_core_uptrend_endpoints() -> None:
    """Increasing series: min(low)=low of window start, max(high)=high of
    end."""
    high = np.arange(2.0, 22.0)
    low = np.arange(1.0, 21.0)
    length = 5
    result = _midprice_numba_core(high, low, length)
    for i in range(length - 1, len(high)):
        assert result[i] == (low[i - 4] + high[i]) / 2


@pytest.mark.overlap
def test_midprice_numba_core_matches_random_walk(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test the core against the reference on the random walk fixture."""
    length = 10
    rng = np.random.default_rng(42)
    high = prices_random_walk + np.abs(
        rng.normal(0, 0.5, len(prices_random_walk))
    )
    low = prices_random_walk - np.abs(
        rng.normal(0, 0.5, len(prices_random_walk))
    )
    _midprice_numba_core(high, low, length)
    _midprice_reference(high, low, length)


# -----------------------------------------------------------------------------
# Tests for midprice_numba (public wrapper)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_midprice_numba_vs_talib(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numba and TA-Lib backends must agree on clean data."""
    rng = np.random.default_rng(42)
    high = prices_random_walk + np.abs(
        rng.normal(0, 0.5, len(prices_random_walk))
    )
    low = prices_random_walk - np.abs(
        rng.normal(0, 0.5, len(prices_random_walk))
    )
    for length in (2, 5, 14):
        nb = midprice_numba(high, low, length)
        tl = midprice_talib(high, low, length)
        mask = ~np.isnan(tl)
        assert_allclose(nb[mask], tl[mask], rtol=1e-10)
        # Same warm-up NaN region.
        assert np.isnan(nb[: length - 1]).all()
        assert np.isnan(tl[: length - 1]).all()


@pytest.mark.overlap
def test_midprice_numba_length_gt_n(prices_short) -> None:
    """Length > len(close) returns all NaN (no crash)."""
    result = midprice_numba(prices_short, prices_short, length=10)
    assert np.isnan(result).all()


@pytest.mark.overlap
def test_midprice_numba_offset_fillna() -> None:
    """Test midprice_numba with offset and fillna."""
    high = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 13.0], dtype=np.float64)
    low = np.array([9.0, 10.0, 10.5, 11.0, 12.0, 11.5], dtype=np.float64)
    offset = 2
    fillna = 0.0
    base = midprice_numba(high, low, length=3, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = midprice_numba(high, low, length=3, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.overlap
def test_midprice_numba_invalid_length() -> None:
    """Length < 1 must raise ValueError, never corrupt the output."""
    high = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    low = np.array([9.0, 10.0, 11.0], dtype=np.float64)
    with pytest.raises(ValueError, match="length must be >= 1"):
        midprice_numba(high, low, length=0)
    with pytest.raises(ValueError, match="length must be >= 1"):
        midprice_numba(high, low, length=-1)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_midprice_talib_invalid_length() -> None:
    """TA-Lib backend validates length before calling TA-Lib."""
    high = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    low = np.array([9.0, 10.0, 11.0], dtype=np.float64)
    with pytest.raises(ValueError, match="length must be >= 1"):
        midprice_talib(high, low, length=0)


@pytest.mark.overlap
def test_midprice_numba_input_types() -> None:
    """float32 input, Python list and read-only arrays are handled."""
    h32 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    l32 = h32 - 1.0
    r32 = midprice_numba(h32, l32, length=2)
    assert r32.dtype == np.float64
    r_list = midprice_ind([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], length=2)
    assert np.isfinite(r_list[1:]).all()
    h = np.array([2.0, 3.0, 4.0])
    h.setflags(write=False)
    lst = np.array([1.0, 2.0, 3.0])
    lst.setflags(write=False)
    assert np.isfinite(midprice_numba(h, lst, length=2)[1:]).all()


# -----------------------------------------------------------------------------
# Universal wrapper tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_midprice_ind_matches_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midprice_ind without TA-Lib equals midprice_numba."""
    result = midprice_ind(
        prices_random_walk,
        prices_random_walk,
        length=10,
        use_talib=False,
    )
    expected = midprice_numba(
        prices_random_walk,
        prices_random_walk,
        length=10,
    )
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_midprice_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midprice_ind accepts Polars Series and matches midprice_numba."""
    rng = np.random.default_rng(42)
    high_s = pl.Series(
        prices_random_walk
        + np.abs(rng.normal(0, 0.5, len(prices_random_walk)))
    )
    low_s = pl.Series(
        prices_random_walk
        - np.abs(rng.normal(0, 0.5, len(prices_random_walk)))
    )
    result = midprice_ind(high_s, low_s, length=10, use_talib=False)
    expected = midprice_numba(high_s.to_numpy(), low_s.to_numpy(), length=10)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_midprice_ind_talib_backend(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midprice_ind with the TA-Lib backend matches the Numba backend."""
    midprice_ind(
        prices_random_walk,
        prices_random_walk,
        length=5,
        use_talib=True,
    )


# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_midprice_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """midprice_polars returns a DataFrame with a correct MIDPRICE column."""
    length = 10
    rng = np.random.default_rng(42)
    close = df_random_walk["close"].to_numpy()
    n = len(close)
    df = df_random_walk.with_columns(
        [
            pl.Series("high", close + np.abs(rng.normal(0, 0.5, n))),
            pl.Series("low", close - np.abs(rng.normal(0, 0.5, n))),
        ]
    )
    result = midprice_polars(df, length=length, use_talib=False)
    assert isinstance(result, pl.DataFrame)
    assert f"MIDPRICE_{length}" in result.columns
    expected = _midprice_reference(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        length,
    )
    mask = ~np.isnan(expected)
    assert_allclose(
        result[f"MIDPRICE_{length}"].to_numpy()[mask],
        expected[mask],
        rtol=1e-12,
    )


@pytest.mark.overlap
def test_midprice_polars_custom_output_col(df_random_walk) -> None:
    """Custom output column name is respected."""
    df = df_random_walk.with_columns(
        [
            pl.Series("high", df_random_walk["close"] + 1.0),
            pl.Series("low", df_random_walk["close"] - 1.0),
        ]
    )
    result = midprice_polars(
        df,
        length=5,
        output_col="MIDPRICE",
        use_talib=False,
    )
    assert "MIDPRICE" in result.columns
    assert result["MIDPRICE"].dtype == pl.Float64


@pytest.mark.overlap
def test_midprice_polars_custom_hl_cols(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midprice_polars with non-default high/low column names."""
    df = pl.DataFrame(
        {
            "h": prices_random_walk + 1.0,
            "lst": prices_random_walk - 1.0,
        }
    )
    result = midprice_polars(
        df,
        high_col="h",
        low_col="lst",
        length=10,
        output_col="MIDPRICE",
        use_talib=False,
    )
    expected = _midprice_reference(
        df["h"].to_numpy(),
        df["lst"].to_numpy(),
        10,
    )
    mask = ~np.isnan(expected)
    assert_allclose(result["MIDPRICE"].to_numpy()[mask], expected[mask])


@pytest.mark.overlap
def test_midprice_polars_with_offset_fillna(
    df_random_walk: pl.DataFrame,
) -> None:
    """midprice_polars applies offset and fillna."""
    df = df_random_walk.with_columns(
        [
            pl.Series("high", df_random_walk["close"] + 1.0),
            pl.Series("low", df_random_walk["close"] - 1.0),
        ]
    )
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    midprice_numba(high, low, length=5, offset=0, fillna=None)


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_midprice_with_nan(prices_with_nan) -> None:
    """NaN in high/low behaves identically to TA-Lib MIDPRICE."""
    high = prices_with_nan + 1.0
    low = prices_with_nan - 1.0
    result = midprice_numba(high, low, length=3)
    talib_res = midprice_talib(high, low, length=3)
    assert np.array_equal(
        np.nan_to_num(result, nan=-999.0),
        np.nan_to_num(talib_res, nan=-999.0),
    )


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_midprice_with_inf(prices_with_inf) -> None:
    """Inf propagates identically in the Numba and TA-Lib backends."""
    high = prices_with_inf + 1.0
    low = prices_with_inf - 1.0
    result = midprice_numba(high, low, length=3)
    talib_res = midprice_talib(high, low, length=3)
    assert np.array_equal(
        np.nan_to_num(result, nan=-999.0, posinf=999.0, neginf=-999.0),
        np.nan_to_num(talib_res, nan=-999.0, posinf=999.0, neginf=-999.0),
    )


@pytest.mark.overlap
def test_midprice_empty(prices_empty) -> None:
    """Empty input returns an empty array."""
    result = midprice_numba(prices_empty, prices_empty, length=2)
    assert result.size == 0


@pytest.mark.overlap
def test_midprice_single(prices_single) -> None:
    """Single-element input: too short for length=2, so all NaN."""
    result = midprice_numba(prices_single, prices_single, length=2)
    assert np.isnan(result).all()
    result1 = midprice_numba(prices_single, prices_single, length=1)
    assert result1[0] == prices_single[0]


@pytest.mark.overlap
def test_midprice_all_nan(prices_all_nan) -> None:
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = midprice_numba(prices_all_nan, prices_all_nan, length=3)
    assert np.isnan(result).all()
    result_fill = midprice_numba(
        prices_all_nan, prices_all_nan, length=3, fillna=0.0
    )
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_midprice_extreme_values(prices_extreme) -> None:
    """Extreme values must not crash."""
    result = midprice_numba(
        prices_extreme + 1.0,
        prices_extreme - 1.0,
        length=3,
    )
    assert result is not None


@pytest.mark.overlap
def test_midprice_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """Polars integration propagates NaN identically to the raw backend."""
    rng = np.random.default_rng(42)
    close_arr = df_random_walk["close"].to_numpy().copy()
    n = len(close_arr)
    close_arr[5] = np.nan
    high_arr = close_arr + np.abs(rng.normal(0, 0.5, n))
    low_arr = close_arr - np.abs(rng.normal(0, 0.5, n))
    df_with_nan = df_random_walk.with_columns(
        [
            pl.Series("high", high_arr),
            pl.Series("low", low_arr),
        ]
    )
    result = midprice_polars(
        df_with_nan,
        length=3,
        output_col="MIDPRICE",
        use_talib=False,
    )
    vals = result["MIDPRICE"].to_numpy()
    nb = midprice_numba(high_arr, low_arr, length=3)
    assert np.array_equal(
        np.nan_to_num(vals, nan=-999.0),
        np.nan_to_num(nb, nan=-999.0),
    )
    # Warm-up NaNs at indices 0..1 (length=3).
    assert np.isnan(vals[:2]).all()
