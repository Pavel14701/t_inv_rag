# -*- coding: utf-8 -*-
"""Unit tests for Volume Weighted Moving Average (VWMA) module.

Tests cover:
- vwma_numpy against the pure Python reference (multiple lengths)
- weight semantics (volume pulls VWMA towards the heavy bar)
- length=1 and constant series
- parameter validation (length, length mismatch, series too short)
- offset and fillna
- NaN handling (nan_policy: raise / ignore / ffill) and Inf -> NaN,
  including identical behaviour of the Numba and TA-Lib backends
- zero-volume windows (documented 0/0 -> NaN)
- vwma_ind with Polars Series and non-contiguous arrays
- vwma_polars DataFrame integration
- IEEE 754 compliance (empty, extreme, all-NaN volume)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_almost_equal

from ta.src.external import talib_available
from ta.src.volume.vwma import vwma_ind, vwma_numpy, vwma_polars


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _vwma_reference(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure Python reference VWMA via explicit window sums."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(length - 1, n):
        w = volume[i - length + 1 : i + 1]
        denom = w.sum()
        if denom == 0:
            continue  # keep NaN for zero-volume windows
        out[i] = np.sum(close[i - length + 1 : i + 1] * w) / denom
    return out


def _prices_and_volume(
    prices_random_walk: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pair the random-walk fixture with a deterministic volume series."""
    volume = np.linspace(1.0, 3.0, len(prices_random_walk))
    return prices_random_walk, volume


# -----------------------------------------------------------------------------
# Correctness tests
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_numpy_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """vwma_numpy must match the pure Python reference for many lengths."""
    close, volume = _prices_and_volume(prices_random_walk)
    for length in (1, 5, 14, 50):
        result = vwma_numpy(close, volume, length=length, use_talib=False)
        expected = _vwma_reference(close, volume, length)
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.volume
def test_vwma_length_one(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """length=1: VWMA equals the close price itself."""
    close, volume = _prices_and_volume(prices_random_walk)
    result = vwma_numpy(close, volume, length=1, use_talib=False)
    assert_allclose(result, close, rtol=1e-12)


@pytest.mark.volume
def test_vwma_constant_close() -> None:
    """Constant close with positive volume: VWMA equals the constant."""
    close = np.full(30, 55.5)
    volume = np.linspace(1.0, 2.0, 30)
    result = vwma_numpy(close, volume, length=10, use_talib=False)
    assert_allclose(result[9:], 55.5, rtol=1e-12)


@pytest.mark.volume
def test_vwma_volume_weighting() -> None:
    """A heavy-volume bar pulls VWMA towards its own close."""
    # Bars: closes 100, 100, 110; volumes 1, 1, 100
    close = np.array([100.0, 100.0, 110.0])
    volume = np.array([1.0, 1.0, 100.0])
    result = vwma_numpy(close, volume, length=3, use_talib=False)
    expected = (100 * 1 + 100 * 1 + 110 * 100) / 102.0
    assert_almost_equal(result[-1], expected, decimal=10)
    # Must be much closer to 110 than the plain SMA (103.33)
    assert result[-1] > 109.0


@pytest.mark.volume
def test_vwma_warmup_nans(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """First length-1 values are NaN, the rest are finite."""
    close, volume = _prices_and_volume(prices_random_walk)
    length = 14
    result = vwma_numpy(close, volume, length=length, use_talib=False)
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()
    assert len(result) == len(close)


# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    volume = np.ones(10)
    for bad_length in (0, -5):
        with pytest.raises(ValueError, match="must be >= 1"):
            vwma_numpy(close, volume, length=bad_length, use_talib=False)


@pytest.mark.volume
def test_vwma_length_mismatch() -> None:
    """Close and volume of different lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        vwma_numpy(
            np.arange(1.0, 21.0), np.ones(10), length=5, use_talib=False
        )


@pytest.mark.volume
def test_vwma_too_short() -> None:
    """Series shorter than `length` raises ValueError, not silent NaN."""
    with pytest.raises(ValueError, match="Input series too short"):
        vwma_numpy(
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            length=5,
            use_talib=False,
        )


# -----------------------------------------------------------------------------
# offset / fillna tests
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_offset(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Positive offset shifts the result forward."""
    close, volume = _prices_and_volume(prices_random_walk)
    base = vwma_numpy(close, volume, length=10, use_talib=False)
    shifted = vwma_numpy(close, volume, length=10, offset=3, use_talib=False)
    assert np.isnan(shifted[:3]).all()
    assert_allclose(shifted[3:], base[:-3], rtol=1e-12, equal_nan=True)


@pytest.mark.volume
def test_vwma_negative_offset(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Negative offset shifts the result backward."""
    close, volume = _prices_and_volume(prices_random_walk)
    base = vwma_numpy(close, volume, length=10, use_talib=False)
    shifted = vwma_numpy(close, volume, length=10, offset=-3, use_talib=False)
    assert np.isnan(shifted[-3:]).all()
    assert_allclose(shifted[:-3], base[3:], rtol=1e-12, equal_nan=True)


@pytest.mark.volume
def test_vwma_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Fillna replaces warm-up NaNs."""
    close, volume = _prices_and_volume(prices_random_walk)
    result = vwma_numpy(close, volume, length=10, fillna=0.0, use_talib=False)
    assert not np.isnan(result).any()
    assert (result[:9] == 0.0).all()


# -----------------------------------------------------------------------------
# NaN / Inf handling (nan_policy)
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_nan_policy_raise() -> None:
    """Default nan_policy='raise' rejects NaN in close or volume."""
    close = np.arange(1.0, 16.0)
    volume = np.ones(15)
    close_with_nan = close.astype(np.float64)
    close_with_nan[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        vwma_numpy(close_with_nan, volume, length=5, use_talib=False)
    volume_with_nan = volume.copy()
    volume_with_nan[6] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        vwma_numpy(close, volume_with_nan, length=5, use_talib=False)


@pytest.mark.volume
def test_vwma_nan_policy_ignore() -> None:
    """nan_policy='ignore' does not raise and lets NaN propagate.

    A NaN close makes only the windows that include it produce NaN.
    (The underlying SMA accumulates, so a NaN in the middle would
    poison all later windows; a trailing NaN only affects the last.)
    """
    close = np.arange(1.0, 16.0)
    volume = np.ones(15)
    close[14] = np.nan  # NaN on the last bar only
    result = vwma_numpy(
        close, volume, length=5, nan_policy="ignore", use_talib=False
    )
    assert np.isnan(result[-1]).all()  # the last window includes the NaN
    # Every window that does not include the NaN is finite
    assert np.isfinite(result[4:-1]).all()


@pytest.mark.volume
def test_vwma_nan_policy_ffill() -> None:
    """nan_policy='ffill' fills the input NaN and yields finite output."""
    close = np.arange(1.0, 16.0)
    volume = np.ones(15)
    close[5] = np.nan
    result = vwma_numpy(
        close, volume, length=5, nan_policy="ffill", use_talib=False
    )
    # First length-1 slots are the warm-up period (no full window yet)
    assert np.isnan(result[:4]).all()
    assert np.isfinite(result[4:]).all()


@pytest.mark.volume
def test_vwma_invalid_nan_policy() -> None:
    """Unknown nan_policy raises ValueError (validated up front)."""
    close = np.arange(1.0, 16.0)
    with pytest.raises(ValueError, match="nan_policy"):
        vwma_numpy(
            close, np.ones(15), length=5, nan_policy="drop", use_talib=False
        )


@pytest.mark.volume
def test_vwma_inf_behaves_like_nan() -> None:
    """Inf in input is replaced with NaN and handled like NaN."""
    close = np.arange(1.0, 16.0)
    volume = np.ones(15)
    close_with_inf = close.astype(np.float64)
    close_with_inf[14] = np.inf  # Inf on the last bar only
    # Default policy raises (Inf became NaN)
    with pytest.raises(ValueError, match="NaN"):
        vwma_numpy(close_with_inf, volume, length=5, use_talib=False)
    # With 'ignore', Inf behaves exactly like NaN (no inf in output)
    result = vwma_numpy(
        close_with_inf, volume, length=5, nan_policy="ignore", use_talib=False
    )
    assert not np.isinf(result).any()
    assert np.isnan(result[-1]).all()
    assert np.isfinite(result[4:-1]).all()


# -----------------------------------------------------------------------------
# Backend parity (Numba vs TA-Lib)
# -----------------------------------------------------------------------------
@pytest.mark.volume
@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
def test_vwma_backend_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numba and TA-Lib backends must agree with the reference."""
    close, volume = _prices_and_volume(prices_random_walk)
    expected = _vwma_reference(close, volume, 14)
    numba_res = vwma_numpy(close, volume, length=14, use_talib=False)
    talib_res = vwma_numpy(close, volume, length=14, use_talib=True)
    assert_allclose(numba_res, expected, rtol=1e-6)
    assert_allclose(talib_res, expected, rtol=1e-6)


@pytest.mark.volume
@pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")
def test_vwma_backend_nan_parity() -> None:
    """Both backends must handle a trailing NaN identically ('ignore')."""
    close = np.arange(1.0, 16.0)
    volume = np.ones(15)
    close[14] = np.nan  # trailing NaN (valid before the last window)
    numba_res = vwma_numpy(
        close, volume, length=5, nan_policy="ignore", use_talib=False
    )
    talib_res = vwma_numpy(
        close, volume, length=5, nan_policy="ignore", use_talib=True
    )
    assert_allclose(numba_res, talib_res, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Zero-volume windows (documented 0/0 -> NaN)
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_zero_volume_window() -> None:
    """A window whose total volume is zero yields NaN."""
    close = np.arange(1.0, 16.0)
    volume = np.ones(15)
    volume[2:7] = 0.0  # all zero inside the length-5 window ending at 6
    result = vwma_numpy(close, volume, length=5, use_talib=False)
    assert np.isnan(result[6]).all()
    # Windows with positive volume remain finite
    assert np.isfinite(result[10:]).all()


# -----------------------------------------------------------------------------
# vwma_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """vwma_ind accepts Polars Series for both price and volume."""
    close, volume = _prices_and_volume(prices_random_walk)
    s_close = pl.Series(close)
    s_volume = pl.Series(volume)
    result = vwma_ind(s_close, s_volume, length=14, use_talib=False)
    expected = _vwma_reference(close, volume, 14)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.volume
def test_vwma_ind_non_contiguous(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Non-contiguous inputs produce the same values as contiguous."""
    close, volume = _prices_and_volume(prices_random_walk)
    buf_c = np.empty(len(close) * 2)
    buf_c[::2] = close
    nc = buf_c[::2]
    buf_v = np.empty(len(volume) * 2)
    buf_v[::2] = volume
    nv = buf_v[::2]
    assert not nc.flags.c_contiguous
    assert not nv.flags.c_contiguous
    result = vwma_ind(nc, nv, length=5, use_talib=False)
    expected = _vwma_reference(close, volume, 5)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


# -----------------------------------------------------------------------------
# vwma_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """vwma_polars adds a column matching the reference."""
    close = df_random_walk["close"].to_numpy()
    volume = np.linspace(1.0, 3.0, len(close))
    df = df_random_walk.with_columns([pl.Series("volume", volume)])
    result_df = vwma_polars(df, length=14, output_col="VWMA", use_talib=False)
    assert "VWMA" in result_df.columns
    assert result_df["VWMA"].dtype == pl.Float64
    assert len(result_df) == len(df)
    expected = _vwma_reference(close, volume, 14)
    assert_allclose(
        result_df["VWMA"].to_numpy(), expected, rtol=1e-10, equal_nan=True
    )


@pytest.mark.volume
def test_vwma_polars_default_output_col() -> None:
    """Default output column name is VWMA_{length}."""
    df = pl.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0, 5.0], "volume": [10, 20, 30, 40, 50]}
    )
    result_df = vwma_polars(df, length=3, use_talib=False)
    assert "VWMA_3" in result_df.columns


@pytest.mark.volume
def test_vwma_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:
    """vwma_polars applies offset and fillna."""
    close = df_random_walk["close"].to_numpy()
    volume = np.linspace(1.0, 3.0, len(close))
    df = df_random_walk.with_columns([pl.Series("volume", volume)])
    result_df = vwma_polars(
        df, length=10, offset=2, fillna=0.0, output_col="VWMA", use_talib=False
    )
    expected = vwma_numpy(
        close, volume, length=10, offset=2, fillna=0.0, use_talib=False
    )
    assert_allclose(
        result_df["VWMA"].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
@pytest.mark.volume
def test_vwma_all_nan_volume() -> None:
    """All-NaN volume: raise by default, all-NaN with 'ignore'."""
    close = np.arange(1.0, 11.0)
    volume = np.full(10, np.nan)
    with pytest.raises(ValueError, match="NaN"):
        vwma_numpy(close, volume, length=5, use_talib=False)
    result = vwma_numpy(
        close, volume, length=5, nan_policy="ignore", use_talib=False
    )
    assert np.isnan(result).all()


@pytest.mark.volume
def test_vwma_empty(prices_empty) -> None:
    """Empty input raises ValueError (series too short)."""
    with pytest.raises(ValueError, match="Input series too short"):
        vwma_numpy(prices_empty, prices_empty, length=5, use_talib=False)


@pytest.mark.volume
def test_vwma_extreme_values(prices_extreme) -> None:
    """Extreme values (1e300, 1e-300) must not crash."""
    volume = np.linspace(1.0, 2.0, len(prices_extreme))
    result = vwma_numpy(
        prices_extreme, volume, length=5, nan_policy="ignore", use_talib=False
    )
    assert result is not None
    assert len(result) == len(prices_extreme)
