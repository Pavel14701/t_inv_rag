# -*- coding: utf-8 -*-
"""Unit tests for OHLC4 module.

Tests cover:
- ohlc4_numpy against a reference ((open + high + low + close) / 4)
- offset and fillna
- ohlc4_ind with Polars Series and list input
- Polars integration (ohlc4_polars)
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.overlap.ohlc4 import ohlc4_ind, ohlc4_numpy, ohlc4_polars


# -----------------------------------------------------------------------------
# Reference implementation
# -----------------------------------------------------------------------------


def _ohlc4_reference(
    open_: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pure Python reference: (open + high + low + close) / 4."""
    return (open_ + high + low + close) / 4.0


def _make_ohlc(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 42,
) -> tuple:
    """Derive consistent open/high/low/close series from a price series."""
    rng = np.random.default_rng(seed)
    n = len(prices_random_walk)
    open_ = prices_random_walk + rng.normal(0, 0.2, n)
    close = prices_random_walk + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
    return open_, high, low, close


# -----------------------------------------------------------------------------
# Basic tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ohlc4_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Compare ohlc4_numpy with the pure-Python reference."""
    o, h, low_, c = _make_ohlc(prices_random_walk)
    result = ohlc4_numpy(o, h, low_, c)
    expected = _ohlc4_reference(o, h, low_, c)
    assert result.shape == o.shape
    assert result.dtype == np.float64
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.overlap
def test_ohlc4_simple_values() -> None:
    """Exact values on a small hand-crafted series."""
    o = np.array([10.0, 11.0, 12.0])
    h = np.array([11.0, 12.0, 13.0])
    low_ = np.array([9.0, 10.0, 11.0])
    c = np.array([10.5, 11.5, 12.5])
    result = ohlc4_numpy(o, h, low_, c)
    assert_allclose(result, [10.125, 11.125, 12.125], rtol=1e-12)


@pytest.mark.overlap
def test_ohlc4_between_low_and_high() -> None:
    """OHLC4 lies within [low, high] for every bar."""
    rng = np.random.default_rng(42)
    n = 100
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    o, h, low_, c = _make_ohlc(close, seed=1)
    result = ohlc4_numpy(o, h, low_, c)
    assert np.all(result >= low_ - 1e-9)
    assert np.all(result <= h + 1e-9)


@pytest.mark.overlap
def test_ohlc4_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna using _apply_offset_fillna."""
    o, h, low_, c = _make_ohlc(prices_random_walk)
    offset = 2
    fillna = 0.0
    base = ohlc4_numpy(o, h, low_, c, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = ohlc4_numpy(o, h, low_, c, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
    assert (result[:offset] == fillna).all()


@pytest.mark.overlap
def test_ohlc4_input_types() -> None:
    """float32 input, Python lists and read-only arrays are handled."""
    o32 = np.array([10.0, 11.0], dtype=np.float32)
    r32 = ohlc4_numpy(o32, o32 + 1, o32 - 1, o32 + 0.5)
    assert r32.dtype == np.float64
    r_list = ohlc4_ind(
        [10.0, 11.0],
        [11.0, 12.0],
        [9.0, 10.0],
        [10.5, 11.5],
    )
    assert_allclose(r_list, [10.125, 11.125], rtol=1e-12)
    o = np.array([10.0, 11.0])
    o.setflags(write=False)
    assert np.isfinite(ohlc4_numpy(o, o + 1, o - 1, o + 0.5)).all()


@pytest.mark.overlap
def test_ohlc4_2d_input_raises() -> None:
    """2D input must not silently produce wrong results."""
    # numba dispatch raises TypeError before the shape validation runs
    with pytest.raises((ValueError, TypeError)):
        ohlc4_numpy(
            np.ones((3, 3)),
            np.ones((3, 3)),
            np.ones((3, 3)),
            np.ones((3, 3)),
        )


# -----------------------------------------------------------------------------
# Universal wrapper tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ohlc4_ind_matches_numpy(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """ohlc4_ind returns the same result as ohlc4_numpy."""
    o, h, low_, c = _make_ohlc(prices_random_walk)
    result = ohlc4_ind(o, h, low_, c)
    expected = ohlc4_numpy(o, h, low_, c)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_ohlc4_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """ohlc4_ind accepts Polars Series and matches ohlc4_numpy."""
    o, h, low_, c = _make_ohlc(prices_random_walk)
    result = ohlc4_ind(
        pl.Series(o), pl.Series(h), pl.Series(low_), pl.Series(c)
    )
    expected = ohlc4_numpy(o, h, low_, c)
    assert_allclose(result, expected, rtol=1e-12)


# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ohlc4_polars_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """ohlc4_polars returns a DataFrame with a correct OHLC4 column."""
    o, h, low_, c = _make_ohlc(prices_random_walk)
    df = pl.DataFrame({"open": o, "high": h, "low": low_, "close": c})
    result = ohlc4_polars(df)
    assert isinstance(result, pl.DataFrame)
    assert "OHLC4" in result.columns
    assert result["OHLC4"].dtype == pl.Float64
    assert_allclose(
        result["OHLC4"].to_numpy(), _ohlc4_reference(o, h, low_, c)
    )


@pytest.mark.overlap
def test_ohlc4_polars_custom_cols(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Non-default column names and custom output column are respected."""
    o, h, low_, c = _make_ohlc(prices_random_walk)
    df = pl.DataFrame({"o": o, "h": h, "low_": low_, "c": c})
    result = ohlc4_polars(
        df,
        open_col="o",
        high_col="h",
        low_col="low_",
        close_col="c",
        output_col="AVG",
    )
    assert "AVG" in result.columns
    assert_allclose(result["AVG"].to_numpy(), _ohlc4_reference(o, h, low_, c))


@pytest.mark.overlap
def test_ohlc4_polars_with_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """ohlc4_polars applies offset and fillna."""
    o, h, low_, c = _make_ohlc(prices_random_walk)
    df = pl.DataFrame({"open": o, "high": h, "low": low_, "close": c})
    offset = 3
    fillna = 0.0
    base = ohlc4_numpy(o, h, low_, c)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = ohlc4_polars(df, offset=offset, fillna=fillna)
    assert_allclose(result["OHLC4"].to_numpy(), expected, rtol=1e-12)
    assert (result["OHLC4"].to_numpy()[:offset] == fillna).all()


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ohlc4_with_nan() -> None:
    """NaN in any component makes that bar's OHLC4 NaN (per-element)."""
    o = np.array([1.0, np.nan, 3.0])
    h = np.array([2.0, 3.0, 4.0])
    low_ = np.array([0.0, 1.0, 2.0])
    c = np.array([1.5, 2.5, 3.5])
    result = ohlc4_numpy(o, h, low_, c)
    assert result[0] == pytest.approx(1.125)
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(3.125)


@pytest.mark.overlap
def test_ohlc4_with_inf() -> None:
    """Inf in input propagates naturally (no crash, no silent replacement)."""
    o = np.array([1.0, np.inf, 3.0])
    h = np.array([2.0, 3.0, 4.0])
    low_ = np.array([0.0, 1.0, 2.0])
    c = np.array([1.5, 2.5, 3.5])
    result = ohlc4_numpy(o, h, low_, c)
    assert result[0] == pytest.approx(1.125)
    assert np.isinf(result[1])
    assert result[2] == pytest.approx(3.125)


@pytest.mark.overlap
def test_ohlc4_empty() -> None:
    """Empty input returns an empty array."""
    e = np.array([], dtype=np.float64)
    result = ohlc4_numpy(e, e, e, e)
    assert result.size == 0


@pytest.mark.overlap
def test_ohlc4_extreme_values() -> None:
    """Extreme values must not crash (mean of 1e300-scale values)."""
    v = np.array([1e300, -1e300, 1e300])
    result = ohlc4_numpy(v, v, v, v)
    assert np.isfinite(result).all()  # (+1e300 * 4) / 4 stays finite
    result_mixed = ohlc4_numpy(v, -v, v, -v)
    assert_allclose(result_mixed, 0.0, atol=1e-290)


@pytest.mark.overlap
def test_ohlc4_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """Polars integration propagates NaN correctly."""
    close = df_random_walk["close"].to_numpy()
    o, h, low_, c = _make_ohlc(close)
    o = o.copy()
    o[5] = np.nan
    df = df_random_walk.with_columns(
        [
            pl.Series("open", o),
            pl.Series("high", h),
            pl.Series("low", low_),
            pl.Series("close", c),
        ]
    )
    result = ohlc4_polars(df)
    vals = result["OHLC4"].to_numpy()
    assert np.isfinite(vals[:5]).all()
    assert np.isnan(vals[5])
    assert np.isfinite(vals[6:]).all()
