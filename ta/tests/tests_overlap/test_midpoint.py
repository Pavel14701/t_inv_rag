# -*- coding: utf-8 -*-
"""Unit tests for MIDPOINT module.

Tests cover:
- Numba core function (_midpoint_numba_core) against a reference
- Numba vs TA-Lib parity on clean data
- Full midpoint_numba with offset and fillna
- Input validation (length < 1)
- Universal wrapper (midpoint_ind) with NumPy, Polars Series and list input
- Polars integration (midpoint_polars)
- IEEE 754 compliance (NaN, Inf, empty, short, all-NaN, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.external import talib_available
from ta.src.overlap.midpoint import (
    _midpoint_numba_core,
    midpoint_ind,
    midpoint_numba,
    midpoint_polars,
    midpoint_talib,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python / numpy)
# -----------------------------------------------------------------------------


def _midpoint_reference(
    close: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure-Python reference: rolling (min + max) / 2."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        window = close[i - length + 1 : i + 1]
        out[i] = (np.min(window) + np.max(window)) * 0.5
    return out


# -----------------------------------------------------------------------------
# Numba core function tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
@pytest.mark.parametrize(
    "length", [2, 3, 5], ids=["len2", "len3", "len5"]
)
def test_midpoint_numba_core_basic(length: int) -> None:
    """Test _midpoint_numba_core against the pure-Python reference."""
    close = np.array(
        [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5],
        dtype=np.float64,
    )
    result = _midpoint_numba_core(close, length)
    expected = _midpoint_reference(close, length)
    assert result.shape == close.shape
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
    # First `length - 1` values are the warm-up NaNs.
    assert np.isnan(result[: length - 1]).all()


@pytest.mark.overlap
def test_midpoint_numba_core_window_bounds() -> None:
    """Midpoint equals (min + max) / 2 of each exact window."""
    close = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    length = 4
    result = _midpoint_numba_core(close, length)
    assert result[3] == (np.min(close[0:4]) + np.max(close[0:4])) / 2
    assert result[4] == (np.min(close[1:5]) + np.max(close[1:5])) / 2
    assert result[7] == (np.min(close[4:8]) + np.max(close[4:8])) * 0.5


@pytest.mark.overlap
def test_midpoint_numba_core_constant_series() -> None:
    """Constant series -> midpoint equals the constant everywhere."""
    close = np.full(20, 42.0)
    result = _midpoint_numba_core(close, 5)
    finite = result[~np.isnan(result)]
    assert finite.size == len(close) - 4
    assert (finite == 42.0).all()


@pytest.mark.overlap
def test_midpoint_numba_core_monotonic_series() -> None:
    """Strictly increasing series: midpoint = average of window endpoints."""
    close = np.arange(1.0, 21.0)
    length = 5
    result = _midpoint_numba_core(close, length)
    for i in range(length - 1, len(close)):
        assert result[i] == close[i - 2]  # (close[i-4] + close[i]) / 2


@pytest.mark.overlap
def test_midpoint_numba_core_matches_random_walk(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test the core against the reference on the random walk fixture."""
    length = 10
    result = _midpoint_numba_core(prices_random_walk, length)
    expected = _midpoint_reference(prices_random_walk, length)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for midpoint_numba (public wrapper)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.parametrize("length", [2, 5, 14], ids=["len2", "len5", "len14"])
@pytest.mark.overlap
def test_midpoint_numba_vs_talib(
    prices_random_walk: npt.NDArray[np.float64],
    length: int,
) -> None:
    """Numba and TA-Lib backends must agree on clean data."""
    close = prices_random_walk
    nb = midpoint_numba(close, length)
    tl = midpoint_talib(close, length)
    mask = ~np.isnan(tl)
    assert_allclose(nb[mask], tl[mask], rtol=1e-10)
    # Same warm-up NaN region.
    assert np.isnan(nb[: length - 1]).all()
    assert np.isnan(tl[: length - 1]).all()


@pytest.mark.overlap
def test_midpoint_numba_length_gt_n(prices_short) -> None:
    """Length > len(close) returns all NaN (no crash)."""
    result = midpoint_numba(prices_short, length=10)
    assert np.isnan(result).all()


@pytest.mark.overlap
def test_midpoint_numba_offset_fillna() -> None:
    """Test midpoint_numba with offset and fillna."""
    close = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 14.0], dtype=np.float64)
    offset = 2
    fillna = 0.0
    base = midpoint_numba(close, length=3, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = midpoint_numba(close, length=3, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.overlap
def test_midpoint_numba_invalid_length() -> None:
    """Length < 1 must raise ValueError, never corrupt the output."""
    close = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    with pytest.raises(ValueError, match="length must be >= 1"):
        midpoint_numba(close, length=0)
    with pytest.raises(ValueError, match="length must be >= 1"):
        midpoint_numba(close, length=-1)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_midpoint_talib_invalid_length() -> None:
    """TA-Lib backend validates length before calling TA-Lib."""
    close = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    with pytest.raises(ValueError, match="length must be >= 1"):
        midpoint_talib(close, length=0)


@pytest.mark.overlap
def test_midpoint_numba_input_types() -> None:
    """float32 input, Python list and read-only arrays are handled."""
    r32 = midpoint_numba(np.array([1.0, 2.0, 3.0], dtype=np.float32), length=2)
    assert r32.dtype == np.float64
    r_list = midpoint_ind([1.0, 2.0, 3.0], length=2)
    assert np.isfinite(r_list[1:]).all()
    c = np.array([1.0, 2.0, 3.0])
    c.setflags(write=False)
    assert np.isfinite(midpoint_numba(c, length=2)[1:]).all()


# -----------------------------------------------------------------------------
# Universal wrapper tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_midpoint_ind_matches_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midpoint_ind without TA-Lib equals midpoint_numba."""
    result = midpoint_ind(prices_random_walk, length=10, use_talib=False)
    expected = midpoint_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_midpoint_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midpoint_ind accepts a Polars Series and matches midpoint_numba."""
    s = pl.Series(prices_random_walk)
    result = midpoint_ind(s, length=10, use_talib=False)
    expected = midpoint_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_midpoint_ind_talib_backend(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midpoint_ind with the TA-Lib backend matches the Numba backend."""
    result = midpoint_ind(prices_random_walk, length=5, use_talib=True)
    expected = midpoint_numba(prices_random_walk, length=5)
    mask = ~np.isnan(expected)
    assert_allclose(result[mask], expected[mask], rtol=1e-10)


# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_midpoint_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """midpoint_polars returns a DataFrame with a correct MIDPOINT column."""
    length = 10
    result = midpoint_polars(df_random_walk, length=length, use_talib=False)
    assert isinstance(result, pl.DataFrame)
    assert f"MIDPOINT_{length}" in result.columns
    close_arr = df_random_walk["close"].to_numpy()
    expected = _midpoint_reference(close_arr, length)
    mask = ~np.isnan(expected)
    assert_allclose(
        result[f"MIDPOINT_{length}"].to_numpy()[mask],
        expected[mask],
        rtol=1e-12,
    )


@pytest.mark.overlap
def test_midpoint_polars_custom_output_col(df_random_walk) -> None:
    """Custom output column name is respected."""
    result = midpoint_polars(
        df_random_walk,
        length=5,
        output_col="MIDPOINT",
        use_talib=False,
    )
    assert "MIDPOINT" in result.columns
    assert result["MIDPOINT"].dtype == pl.Float64


@pytest.mark.overlap
def test_midpoint_polars_custom_close_col(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """midpoint_polars with a non-default close column name."""
    df = pl.DataFrame({"price": prices_random_walk})
    result = midpoint_polars(
        df,
        close_col="price",
        length=10,
        output_col="MIDPOINT",
        use_talib=False,
    )
    expected = _midpoint_reference(prices_random_walk, 10)
    mask = ~np.isnan(expected)
    assert_allclose(result["MIDPOINT"].to_numpy()[mask], expected[mask])


@pytest.mark.overlap
def test_midpoint_polars_with_offset_fillna(
    df_random_walk: pl.DataFrame,
) -> None:
    """midpoint_polars applies offset and fillna."""
    offset = 2
    fillna = 0.0
    close_arr = df_random_walk["close"].to_numpy()
    base = midpoint_numba(close_arr, length=5, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = midpoint_polars(
        df_random_walk,
        length=5,
        offset=offset,
        fillna=fillna,
        output_col="MIDPOINT",
        use_talib=False,
    )
    assert_allclose(result["MIDPOINT"].to_numpy(), expected, rtol=1e-12)


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_midpoint_with_nan(prices_with_nan) -> None:
    """NaN windows skip the NaN (same behaviour as TA-Lib MIDPOINT)."""
    result = midpoint_numba(prices_with_nan, length=3)
    talib_res = midpoint_talib(prices_with_nan, length=3)
    mask = ~np.isnan(talib_res)
    assert_allclose(result[mask], talib_res[mask], rtol=1e-12)


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_midpoint_with_inf(prices_with_inf) -> None:
    """Inf propagates through min/max identically in both backends."""
    result = midpoint_numba(prices_with_inf, length=3)
    talib_res = midpoint_talib(prices_with_inf, length=3)
    # IEEE 754 parity: both backends produce the exact same output.
    assert np.array_equal(
        np.nan_to_num(result, nan=-999.0, posinf=999.0, neginf=-999.0),
        np.nan_to_num(talib_res, nan=-999.0, posinf=999.0, neginf=-999.0),
    )
    # Inf is carried into the window's max (IEEE 754 semantics).
    assert result[5] == np.inf and result[6] == np.inf
    assert np.isfinite(result[2:5]).all()


@pytest.mark.overlap
def test_midpoint_empty(prices_empty) -> None:
    """Empty input returns an empty array."""
    result = midpoint_numba(prices_empty, length=2)
    assert result.size == 0


@pytest.mark.overlap
def test_midpoint_single(prices_single) -> None:
    """Single-element input: too short for length=2, so all NaN."""
    result = midpoint_numba(prices_single, length=2)
    assert np.isnan(result).all()
    result1 = midpoint_numba(prices_single, length=1)
    assert result1[0] == prices_single[0]


@pytest.mark.overlap
def test_midpoint_all_nan(prices_all_nan) -> None:
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = midpoint_numba(prices_all_nan, length=3)
    assert np.isnan(result).all()
    result_fill = midpoint_numba(prices_all_nan, length=3, fillna=0.0)
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_midpoint_extreme_values(prices_extreme) -> None:
    """Extreme values must not crash."""
    result = midpoint_numba(prices_extreme, length=3)
    assert result is not None


@pytest.mark.overlap
def test_midpoint_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = midpoint_polars(
        df_with_nan,
        length=3,
        output_col="MIDPOINT",
        use_talib=False,
    )
    vals = result["MIDPOINT"].to_numpy()
    # Warm-up NaNs at indices 0..1 (length=3).
    assert np.isnan(vals[:2]).all()
    # Parity with the Numba backend on the same NaN-poisoned input.
    nb_vals = midpoint_numba(close_arr, 3)
    assert np.array_equal(
        np.nan_to_num(vals, nan=-999.0),
        np.nan_to_num(nb_vals, nan=-999.0),
    )
    assert np.array_equal(
        np.nan_to_num(vals, nan=-999.0),
        np.nan_to_num(nb_vals, nan=-999.0),
    )
