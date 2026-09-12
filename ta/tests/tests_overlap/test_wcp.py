# -*- coding: utf-8 -*-
"""Unit tests for Weighted Closing Price (WCP) module.

Tests cover:
- Formula parity: (high + low + 2*close) / 4
- Backend parity: numba vs TA-Lib (WCLPRICE)
- Contiguous / read-only / non-contiguous input handling
- offset and fillna behaviour
- wcp_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme values)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from ta.src.external import talib_available
from ta.src.overlap.wcp import wcp_ind, wcp_numba, wcp_polars, wcp_talib


# -----------------------------------------------------------------------------
# Reference implementation (pure numpy)
# -----------------------------------------------------------------------------


def _wcp_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pure numpy reference: (high + low + 2*close) / 4."""
    return (high + low + 2.0 * close) * 0.25


# -----------------------------------------------------------------------------
# Formula parity
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_wcp_numba_matches_formula(df_ohlc: pl.DataFrame) -> None:
    """Test wcp_numba against the closed-form reference."""
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()

    result = wcp_numba(high, low, close)
    expected = _wcp_reference(high, low, close)
    assert_allclose(result, expected, rtol=0, atol=1e-12)
    assert result.dtype == np.float64
    assert len(result) == len(close)


@pytest.mark.overlap
def test_wcp_no_warmup() -> None:
    """WCP is a pointwise formula: no warm-up NaNs for finite input."""
    high = np.array([2.0, 3.0, 4.0])
    low = np.array([1.0, 2.0, 3.0])
    close = np.array([1.5, 2.5, 3.5])
    result = wcp_numba(high, low, close)
    assert not np.isnan(result).any()


@pytest.mark.overlap
def test_wcp_constant_series() -> None:
    """Constant prices give exactly the same constant."""
    c = np.full(20, 50.0)
    result = wcp_numba(c, c, c)
    assert_allclose(result, 50.0, rtol=0, atol=0)


# -----------------------------------------------------------------------------
# Backend parity
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
@pytest.mark.overlap
def test_wcp_backend_parity(df_ohlc: pl.DataFrame) -> None:
    """Numba and TA-Lib backends agree bit-for-bit on finite input."""
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()

    r_numba = wcp_numba(high, low, close)
    r_talib = wcp_talib(high, low, close)
    assert_array_equal(r_numba, r_talib)


@pytest.mark.overlap
def test_wcp_ind_use_talib_false_matches_numba(df_ohlc: pl.DataFrame) -> None:
    """wcp_ind with use_talib=False equals wcp_numba."""
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()

    r_ind = wcp_ind(high, low, close, use_talib=False)
    r_numba = wcp_numba(high, low, close)
    assert_array_equal(r_ind, r_numba)


# -----------------------------------------------------------------------------
# Contiguity / read-only inputs
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_wcp_non_contiguous_input(df_ohlc: pl.DataFrame) -> None:
    """Non-contiguous input gives identical results (regression: the no-op
    contiguous loop previously discarded the copy).
    """
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()

    r_cont = wcp_numba(high, low, close)
    r_strided = wcp_numba(high[::2], low[::2], close[::2])
    assert_allclose(r_strided, r_cont[::2], rtol=0, atol=1e-12)


@pytest.mark.overlap
def test_wcp_readonly_input(df_ohlc: pl.DataFrame) -> None:
    """Read-only (polars) input is accepted."""
    r_series = wcp_numba(
        df_ohlc["high"].to_numpy(),
        df_ohlc["low"].to_numpy(),
        df_ohlc["close"].to_numpy(),
    )
    r_pl = wcp_ind(
        pl.Series(df_ohlc["high"]),
        pl.Series(df_ohlc["low"]),
        pl.Series(df_ohlc["close"]),
        use_talib=False,
    )
    assert_array_equal(r_series, r_pl)


@pytest.mark.overlap
def test_wcp_ind_polars_series_input(df_ohlc: pl.DataFrame) -> None:
    """wcp_ind accepts pl.Series directly."""
    r = wcp_ind(
        df_ohlc["high"], df_ohlc["low"], df_ohlc["close"], use_talib=False
    )
    expected = _wcp_reference(
        df_ohlc["high"].to_numpy(),
        df_ohlc["low"].to_numpy(),
        df_ohlc["close"].to_numpy(),
    )
    assert_allclose(r, expected, rtol=0, atol=1e-12)


# -----------------------------------------------------------------------------
# Offset / fillna
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_wcp_offset(df_ohlc: pl.DataFrame) -> None:
    """Positive offset shifts the series forward."""
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()

    r0 = wcp_numba(high, low, close)
    r2 = wcp_numba(high, low, close, offset=2)
    assert np.isnan(r2[:2]).all()
    assert_allclose(r2[2:], r0[:-2], rtol=0, atol=1e-12)


@pytest.mark.overlap
def test_wcp_fillna(df_ohlc: pl.DataFrame) -> None:
    """Fillna passthrough: no natural NaNs, so values stay unchanged."""
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()

    r = wcp_numba(high, low, close, fillna=-1.0)
    assert not np.isnan(r).any()
    assert_allclose(r, _wcp_reference(high, low, close), rtol=0, atol=1e-12)


@pytest.mark.overlap
def test_wcp_offset_with_fillna(df_ohlc: pl.DataFrame) -> None:
    """Offset-shifted-in NaNs are replaced by fillna."""
    high = df_ohlc["high"].to_numpy()
    low = df_ohlc["low"].to_numpy()
    close = df_ohlc["close"].to_numpy()

    r = wcp_numba(high, low, close, offset=3, fillna=0.0)
    assert (r[:3] == 0.0).all()
    assert not np.isnan(r).any()


# -----------------------------------------------------------------------------
# Polars integration
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_wcp_polars_default_column(df_ohlc: pl.DataFrame) -> None:
    """wcp_polars adds a 'WCP' column."""
    out = wcp_polars(df_ohlc, use_talib=False)
    assert "WCP" in out.columns
    expected = _wcp_reference(
        df_ohlc["high"].to_numpy(),
        df_ohlc["low"].to_numpy(),
        df_ohlc["close"].to_numpy(),
    )
    assert_allclose(out["WCP"].to_numpy(), expected, rtol=0, atol=1e-12)
    assert len(out) == len(df_ohlc)


@pytest.mark.overlap
def test_wcp_polars_custom_column_and_params(df_ohlc: pl.DataFrame) -> None:
    """Custom output column, offset and fillna are honoured."""
    out = wcp_polars(
        df_ohlc,
        offset=2,
        fillna=-9.0,
        use_talib=False,
        output_col="WCP_CUSTOM",
    )
    assert "WCP_CUSTOM" in out.columns
    col = out["WCP_CUSTOM"].to_numpy()
    assert (col[:2] == -9.0).all()


# -----------------------------------------------------------------------------
# IEEE 754 edge cases
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_wcp_nan_propagates_pointwise() -> None:
    """A NaN in any component poisons only the same bar."""
    n = 10
    high = np.arange(float(n))
    low = np.arange(float(n))
    close = np.arange(float(n))
    high[5] = np.nan
    result = wcp_numba(high, low, close)
    assert np.isnan(result[5])
    assert not np.isnan(np.delete(result, 5)).any()


@pytest.mark.overlap
def test_wcp_inf_propagates() -> None:
    """Inf propagates to the same bar only."""
    high = np.arange(10.0)
    high[3] = np.inf
    result = wcp_numba(high, np.arange(10.0), np.arange(10.0))
    assert np.isinf(result[3])
    assert np.isfinite(np.delete(result, 3)).all()


@pytest.mark.overlap
def test_wcp_empty_input() -> None:
    """Empty input yields empty output."""
    empty = np.array([], dtype=np.float64)
    result = wcp_numba(empty, empty, empty)
    assert len(result) == 0


@pytest.mark.overlap
def test_wcp_extreme_values() -> None:
    """Extreme magnitudes do not corrupt other bars."""
    high = np.full(8, 1.0)
    low = np.full(8, 1.0)
    close = np.full(8, 1.0)
    high[0] = 1e300
    result = wcp_numba(high, low, close)
    # (1e300 + 1 + 2) / 4 = 2.5e299: finite, no overflow
    assert result[0] == 2.5e299
    assert_allclose(result[1:], 1.0, rtol=0, atol=0)
