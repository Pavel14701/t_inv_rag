# -*- coding: utf-8 -*-
"""Unit tests for Symmetric Weighted Moving Average (SWMA) module.

Tests cover:
- Symmetric weight generation (even/odd lengths, normalization, caching)
- swma_numba against a pure Python reference
- Warm-up NaN window (length - 1 bars)
- Validation of `length`
- offset and fillna behaviour
- swma_ind with Polars Series / swma_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, short input)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.overlap.swma import (
    _symmetric_weights,
    swma_ind,
    swma_numba,
    swma_polars,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------


def _swma_reference(
    close: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure Python SWMA reference using the module's normalized weights."""
    w = _symmetric_weights(length)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += close[i - j] * w[length - 1 - j]
        out[i] = acc
    return out


# -----------------------------------------------------------------------------
# Weight generation
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_swma_weights_even_length() -> None:
    """Even length 4 gives symmetric triangle [1,2,2,1]/6."""
    w = _symmetric_weights(4)
    expected = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
    assert_allclose(w, expected, rtol=0, atol=1e-15)


@pytest.mark.overlap
def test_swma_weights_odd_length() -> None:
    """Odd length 5 gives symmetric triangle [1,2,3,2,1]/9."""
    w = _symmetric_weights(5)
    assert_allclose(
        w, np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0, rtol=0, atol=1e-15
    )


@pytest.mark.overlap
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 10, 11])
def test_swma_weights_normalized_and_symmetric(length: int) -> None:
    """Weights sum to 1 and are palindromic for all lengths."""
    w = _symmetric_weights(length)
    assert len(w) == length
    assert_allclose(w.sum(), 1.0, rtol=0, atol=1e-12)
    assert_allclose(w, w[::-1], rtol=0, atol=0)


@pytest.mark.overlap
def test_swma_weights_cached() -> None:
    """Weights are cached (lru_cache returns the same object)."""
    assert _symmetric_weights(10) is _symmetric_weights(10)


# -----------------------------------------------------------------------------
# Core calculation
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_swma_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test swma_numba against the pure Python reference."""
    close = prices_random_walk
    length = 10
    result = swma_numba(close, length=length)
    expected = _swma_reference(close, length)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.overlap
def test_swma_warmup_nan_window(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """First length-1 values are NaN, the rest are finite."""
    close = prices_random_walk
    length = 10
    result = swma_numba(close, length=length)
    assert np.isnan(result[: length - 1]).all()
    assert not np.isnan(result[length - 1 :]).any()


@pytest.mark.overlap
def test_swma_shorter_than_length(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Series shorter than length gives all-NaN output."""
    close = prices_random_walk[:5]
    result = swma_numba(close, length=10)
    assert np.isnan(result).all()
    assert len(result) == 5


@pytest.mark.overlap
def test_swma_constant_input() -> None:
    """Constant input is reproduced exactly after warm-up (weights sum 1)."""
    c = np.full(30, 42.0)
    result = swma_numba(c, length=10)
    assert_allclose(result[9:], 42.0, rtol=0, atol=1e-12)


@pytest.mark.overlap
def test_swma_length_one() -> None:
    """length=1 reduces to the input itself (weight [1.0])."""
    close = np.arange(1.0, 11.0)
    result = swma_numba(close, length=1)
    assert_allclose(result, close, rtol=0, atol=0)


@pytest.mark.overlap
@pytest.mark.parametrize("length", [0, -3])
def test_swma_invalid_length(length: int) -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        swma_numba(close, length=length)


# -----------------------------------------------------------------------------
# Contiguity / read-only inputs
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_swma_non_contiguous_input(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided input is accepted and matches the reference on that data
    (a windowed indicator on resampled data is not the resample of the
    full-series indicator, so compare against its own reference).
    """
    close = prices_random_walk[::2]
    r_strided = swma_numba(close, length=10)
    expected = _swma_reference(close, length=10)
    assert_allclose(r_strided, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.overlap
def test_swma_ind_polars_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """swma_ind accepts pl.Series (read-only numpy underneath)."""
    close = prices_random_walk
    r = swma_ind(pl.Series(close), length=10)
    expected = swma_numba(close, length=10)
    assert_allclose(r, expected, rtol=0, atol=0)


# -----------------------------------------------------------------------------
# Offset / fillna
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_swma_offset(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Positive offset shifts the series forward, warm-up zone becomes NaN."""
    close = prices_random_walk
    length = 10
    r0 = swma_numba(close, length=length)
    r2 = swma_numba(close, length=length, offset=2)
    assert np.isnan(r2[:2]).all()
    assert_allclose(r2[2:], r0[:-2], rtol=0, atol=0)


@pytest.mark.overlap
def test_swma_fillna_warmup_only(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Fillna replaces only the warm-up NaNs, valid zone is untouched."""
    close = prices_random_walk
    length = 10
    r = swma_numba(close, length=length, fillna=-1.0)
    assert not np.isnan(r).any()
    assert (r[: length - 1] == -1.0).all()
    expected = swma_numba(close, length=length)
    assert_allclose(r[length - 1 :], expected[length - 1 :], rtol=0, atol=0)


@pytest.mark.overlap
def test_swma_offset_with_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset NaNs are replaced by fillna."""
    close = prices_random_walk
    r = swma_numba(close, length=10, offset=3, fillna=0.0)
    assert (r[:3] == 0.0).all()
    assert not np.isnan(r).any()


# -----------------------------------------------------------------------------
# Polars integration
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_swma_polars_default_column(df_random_walk: pl.DataFrame) -> None:
    """swma_polars adds an 'SWMA_{length}' column by default."""
    out = swma_polars(df_random_walk, length=10)
    assert "SWMA_10" in out.columns
    expected = _swma_reference(df_random_walk["close"].to_numpy(), length=10)
    assert_allclose(
        out["SWMA_10"].to_numpy(), expected, rtol=1e-12, atol=1e-12
    )
    assert len(out) == len(df_random_walk)


@pytest.mark.overlap
def test_swma_polars_custom_column(df_random_walk: pl.DataFrame) -> None:
    """Custom output column name is honoured."""
    out = swma_polars(df_random_walk, length=5, output_col="MY_SWMA")
    assert "MY_SWMA" in out.columns
    assert "SWMA_5" not in out.columns


# -----------------------------------------------------------------------------
# IEEE 754 edge cases
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_swma_nan_poisons_window_then_recovers() -> None:
    """A NaN poisons its full window, later bars recover."""
    close = np.arange(1.0, 16.0)
    close[5] = np.nan
    length = 4
    result = swma_numba(close, length=length)
    # warm-up bars 0..2 stay NaN; windows containing index 5 are
    # i in {5..8}; bars 3..4 are finite
    assert np.isnan(result[:3]).all()
    assert not np.isnan(result[3:5]).any()
    assert np.isnan(result[5:9]).all()
    assert not np.isnan(result[9:]).any()


@pytest.mark.overlap
def test_swma_inf_propagates_in_window() -> None:
    """Inf poisons its window like NaN."""
    close = np.ones(12)
    close[7] = np.inf
    result = swma_numba(close, length=3)
    assert np.isinf(result[7:10]).all()
    assert np.isfinite(result[10:]).all()
    assert np.isfinite(result[2:7]).all()


@pytest.mark.overlap
def test_swma_empty_input() -> None:
    """Empty input yields empty output."""
    empty = np.array([], dtype=np.float64)
    result = swma_numba(empty, length=10)
    assert len(result) == 0
