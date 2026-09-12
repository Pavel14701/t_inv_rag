# -*- coding: utf-8 -*-
"""Unit tests for Sine Weighted Moving Average (SINWMA) module.

Tests cover:
- Weight generation (_sine_weights): normalisation, symmetry, caching,
  read-only protection, invalid length
- _sinwma_numba_core against reference implementation
- sinwma_numba: warm-up NaNs, constant series, offset and fillna,
  parameter validation (length, series too short)
- NaN handling (nan_policy: raise / ignore / ffill) and Inf -> NaN
- sinwma_ind with Polars Series
- sinwma_polars DataFrame integration (default and custom output col)
- IEEE 754 compliance (NaN, Inf, empty, extreme, all-NaN)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_almost_equal

from ta.src.overlap.sinwma import (
    _sine_weights,
    _sinwma_numba_core,
    sinwma_ind,
    sinwma_numba,
    sinwma_polars,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _sinwma_reference(
    close: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure Python reference SINWMA (sine weights via np.convolve)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    w = np.sin(np.arange(1, length + 1) * np.pi / (length + 1))
    w /= w.sum()
    # Most recent bar gets w[-1]; reverse for np.convolve ('valid' mode)
    out[length - 1 :] = np.convolve(close, w[::-1], mode="valid")
    return out


# -----------------------------------------------------------------------------
# Tests for weight generation
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_sinwma_weights_sum_to_one() -> None:
    """Generated sine weights must be normalised to 1."""
    for length in (1, 2, 5, 14, 50):
        weights = _sine_weights(length)
        assert_almost_equal(weights.sum(), 1.0, decimal=6)


@pytest.mark.overlap
def test_sinwma_weights_symmetric() -> None:
    """Sine weights are symmetric by design (Everget formula)."""
    weights = _sine_weights(10)
    assert_allclose(weights, weights[::-1], rtol=1e-12)


@pytest.mark.overlap
def test_sinwma_weights_formula() -> None:
    """Weights must equal sin(i * pi / (length + 1)) normalised."""
    length = 7
    expected = np.sin(np.arange(1, length + 1) * np.pi / (length + 1))
    expected /= expected.sum()
    assert_allclose(_sine_weights(length), expected, rtol=1e-12)


@pytest.mark.overlap
def test_sinwma_weights_cache() -> None:
    """Weights are cached (lru_cache returns the same object)."""
    w1 = _sine_weights(14)
    w2 = _sine_weights(14)
    assert w1 is w2


@pytest.mark.overlap
def test_sinwma_weights_readonly() -> None:
    """Cached weights must be read-only to protect the lru_cache."""
    weights = _sine_weights(10)
    assert not weights.flags.writeable
    with pytest.raises(ValueError):
        weights[0] = 1.0


@pytest.mark.overlap
def test_sinwma_weights_invalid_length() -> None:
    """Length below 1 raises ValueError (no silent NaN weights)."""
    for bad_length in (0, -1, -10):
        with pytest.raises(ValueError, match="must be >= 1"):
            _sine_weights(bad_length)


# -----------------------------------------------------------------------------
# Tests for _sinwma_numba_core
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_sinwma_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """_sinwma_numba_core must match the pure Python reference."""
    for length in (1, 4, 14, 30):
        result = _sinwma_numba_core(prices_random_walk, _sine_weights(length))
        expected = _sinwma_reference(prices_random_walk, length)
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.overlap
def test_sinwma_core_empty() -> None:
    """Empty input returns empty array (no IndexError)."""
    result = _sinwma_numba_core(np.array([]), _sine_weights(5))
    assert result.size == 0


# -----------------------------------------------------------------------------
# Tests for sinwma_numba
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_sinwma_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """sinwma_numba must match the pure Python reference."""
    result = sinwma_numba(prices_random_walk, length=14)
    expected = _sinwma_reference(prices_random_walk, 14)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.overlap
def test_sinwma_numba_invalid_length() -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 21.0)
    for bad_length in (0, -5):
        with pytest.raises(ValueError, match="must be >= 1"):
            sinwma_numba(close, length=bad_length)


@pytest.mark.overlap
def test_sinwma_numba_too_short() -> None:
    """Series shorter than `length` raises ValueError, not silent NaN."""
    with pytest.raises(ValueError, match="Input series too short"):
        sinwma_numba(np.array([1.0, 2.0]), length=5)


@pytest.mark.overlap
def test_sinwma_numba_length_one(prices_short) -> None:
    """length=1 is valid: SINWMA equals the input itself."""
    result = sinwma_numba(prices_short, length=1)
    assert_allclose(result, prices_short, rtol=1e-12)


@pytest.mark.overlap
def test_sinwma_numba_warmup_nans(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """First length-1 values are NaN, the rest are finite."""
    length = 14
    result = sinwma_numba(prices_random_walk, length=length)
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()
    assert len(result) == len(prices_random_walk)


@pytest.mark.overlap
def test_sinwma_numba_constant_series() -> None:
    """SINWMA of a constant series equals the constant."""
    close = np.full(30, 123.456)
    result = sinwma_numba(close, length=10)
    assert_allclose(result[9:], 123.456, rtol=1e-12)


@pytest.mark.overlap
def test_sinwma_numba_sine_weight_profile() -> None:
    """Middle bars weigh more than recent/oldest bars (sine profile)."""
    length = 5
    w = _sine_weights(length)
    assert w[0] < w[length // 2]
    assert w[-1] < w[length // 2]


@pytest.mark.overlap
def test_sinwma_numba_offset(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Positive offset shifts result forward."""
    base = sinwma_numba(prices_random_walk, length=10)
    shifted = sinwma_numba(prices_random_walk, length=10, offset=3)
    assert np.isnan(shifted[:3]).all()
    assert_allclose(shifted[3:], base[:-3], rtol=1e-12, equal_nan=True)


@pytest.mark.overlap
def test_sinwma_numba_negative_offset(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Negative offset shifts result backward."""
    base = sinwma_numba(prices_random_walk, length=10)
    shifted = sinwma_numba(prices_random_walk, length=10, offset=-3)
    assert np.isnan(shifted[-3:]).all()
    assert_allclose(shifted[:-3], base[3:], rtol=1e-12, equal_nan=True)


@pytest.mark.overlap
def test_sinwma_numba_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Fillna replaces warm-up NaNs."""
    result = sinwma_numba(prices_random_walk, length=10, fillna=0.0)
    assert not np.isnan(result).any()
    assert (result[:9] == 0.0).all()


# -----------------------------------------------------------------------------
# NaN / Inf handling (nan_policy)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_sinwma_numba_nan_policy_raise(prices_with_nan) -> None:
    """Default nan_policy='raise' rejects input containing NaN."""
    with pytest.raises(ValueError, match="NaN"):
        sinwma_numba(prices_with_nan, length=5)


@pytest.mark.overlap
def test_sinwma_numba_nan_policy_ignore(prices_with_nan) -> None:
    """nan_policy='ignore' lets NaN propagate within affected windows."""
    result = sinwma_numba(prices_with_nan, length=5, nan_policy="ignore")
    # NaN at index 5 affects windows ending at indices 5..9
    assert np.isnan(result[5:10]).all()
    # Windows without the NaN stay finite
    assert np.isfinite(result[10:]).all()


@pytest.mark.overlap
def test_sinwma_numba_nan_policy_ffill(prices_with_nan) -> None:
    """nan_policy='ffill' fills the input NaN and yields finite output."""
    result = sinwma_numba(prices_with_nan, length=5, nan_policy="ffill")
    # First length-1 slots are the warm-up period (no full window yet)
    assert np.isnan(result[:4]).all()
    # From the first full window on, everything is finite
    assert np.isfinite(result[4:]).all()


@pytest.mark.overlap
def test_sinwma_numba_invalid_nan_policy() -> None:
    """Unknown nan_policy raises ValueError."""
    close = np.arange(1.0, 21.0)
    with pytest.raises(ValueError, match="nan_policy"):
        sinwma_numba(close, length=5, nan_policy="drop")


@pytest.mark.overlap
def test_sinwma_numba_inf_replaced_with_nan(prices_with_inf) -> None:
    """Inf is replaced with NaN and handled like NaN."""
    # Input contains Inf -> default policy raises (Inf became NaN)
    with pytest.raises(ValueError, match="NaN"):
        sinwma_numba(prices_with_inf, length=5)
    # With 'ignore', Inf behaves exactly like NaN (no inf in output)
    result = sinwma_numba(prices_with_inf, length=5, nan_policy="ignore")
    assert np.isnan(result[5:10]).all()
    assert not np.isinf(result).any()
    assert np.isfinite(result[10:]).all()


# -----------------------------------------------------------------------------
# Tests for sinwma_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_sinwma_ind_with_ndarray(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """sinwma_ind with ndarray input matches reference."""
    result = sinwma_ind(prices_random_walk, length=14)
    expected = _sinwma_reference(prices_random_walk, 14)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.overlap
def test_sinwma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """sinwma_ind accepts a Polars Series."""
    s = pl.Series(prices_random_walk)
    result = sinwma_ind(s, length=14)
    expected = _sinwma_reference(prices_random_walk, 14)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.overlap
def test_sinwma_ind_non_contiguous() -> None:
    """Non-contiguous input produces the same values as contiguous."""
    close = np.arange(1.0, 41.0)
    # Interleave into a (40, 2) array and take every second element:
    # yields a strided (non-contiguous) view holding the same values
    interleaved = np.empty((40, 2))
    interleaved[:, 0] = close
    interleaved[:, 1] = close
    view = interleaved[:, 0]
    assert not view.flags.c_contiguous
    result = sinwma_ind(view, length=5)
    expected = sinwma_ind(close, length=5)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for sinwma_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_sinwma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """sinwma_polars adds a column matching the reference."""
    result_df = sinwma_polars(df_random_walk, length=14, output_col="SINWMA")
    assert "SINWMA" in result_df.columns
    assert result_df["SINWMA"].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk["close"].to_numpy()
    expected = _sinwma_reference(close_arr, 14)
    assert_allclose(
        result_df["SINWMA"].to_numpy(), expected, rtol=1e-10, equal_nan=True
    )


@pytest.mark.overlap
def test_sinwma_polars_default_output_col() -> None:
    """Default output column name is SINWMA_{length}."""
    df = pl.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    result_df = sinwma_polars(df, length=5)
    assert "SINWMA_5" in result_df.columns


@pytest.mark.overlap
def test_sinwma_polars_offset_fillna(
    df_random_walk: pl.DataFrame,
) -> None:
    """sinwma_polars applies offset and fillna."""
    result_df = sinwma_polars(
        df_random_walk,
        length=10,
        offset=2,
        fillna=0.0,
        output_col="SINWMA",
    )
    close_arr = df_random_walk["close"].to_numpy()
    expected = sinwma_numba(close_arr, length=10, offset=2, fillna=0.0)
    assert_allclose(
        result_df["SINWMA"].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


@pytest.mark.overlap
def test_sinwma_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """sinwma_polars propagates NaN correctly with nan_policy='ignore'."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([pl.Series("close", close_arr)])
    result_df = sinwma_polars(
        df_with_nan, length=5, output_col="SINWMA", nan_policy="ignore"
    )
    sinwma_vals = result_df["SINWMA"].to_numpy()
    assert np.isnan(sinwma_vals[5:10]).all()
    assert np.isfinite(sinwma_vals[10:]).all()


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_sinwma_numba_all_nan(prices_all_nan) -> None:
    """All-NaN input: raise by default, all-NaN with 'ignore'."""
    with pytest.raises(ValueError, match="NaN"):
        sinwma_numba(prices_all_nan, length=5)
    result = sinwma_numba(prices_all_nan, length=5, nan_policy="ignore")
    assert np.isnan(result).all()
    result_fill = sinwma_numba(
        prices_all_nan, length=5, fillna=0.0, nan_policy="ignore"
    )
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_sinwma_numba_empty(prices_empty) -> None:
    """Empty input raises ValueError (series too short)."""
    with pytest.raises(ValueError, match="Input series too short"):
        sinwma_numba(prices_empty, length=5)


@pytest.mark.overlap
def test_sinwma_numba_extreme_values(prices_extreme) -> None:
    """Extreme values (1e300, 1e-300) must not crash."""
    result = sinwma_numba(prices_extreme, length=5, nan_policy="ignore")
    assert result is not None
    assert len(result) == len(prices_extreme)
