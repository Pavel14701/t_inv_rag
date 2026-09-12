# -*- coding: utf-8 -*-
"""Unit tests for Ehlers 3-Pole Super Smoother Filter (SSF3) module.

Tests cover:
- ssf3_numba against a pure Python reference of the same recurrence
- Unit DC gain: constant input reproduced exactly
- Seed semantics for n < 4 and empty input (regression: out-of-bounds write)
- Validation of `length`
- offset and fillna behaviour
- ssf3_ind with Polars Series / ssf3_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.overlap.ssf3 import ssf3_ind, ssf3_numba, ssf3_polars


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------


def _ssf3_reference(
    close: npt.NDArray[np.float64],
    length: int,
    pi: float = 3.14159,
    sqrt3: float = 1.732,
) -> npt.NDArray[np.float64]:
    """Pure Python 3-pole Super Smoother (Everget variant)."""
    n = len(close)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[0] = close[0]
    if n > 1:
        out[1] = close[1]
    if n > 2:
        out[2] = close[2]
    if n < 4:
        return out
    a = np.exp(-pi / length)
    b = 2.0 * a * np.cos(-pi * sqrt3 / length)
    c = a * a
    d4 = c * c
    d3 = -c * (1.0 + b)
    d2 = b + c
    d1 = 1.0 - d2 - d3 - d4
    for i in range(3, n):
        out[i] = (
            d1 * close[i] + d2 * out[i - 1] + d3 * out[i - 2] + d4 * out[i - 3]
        )
    return out


# -----------------------------------------------------------------------------
# Core calculation
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf3_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test ssf3_numba against the pure Python reference."""
    close = prices_random_walk
    result = ssf3_numba(close, length=20)
    expected = _ssf3_reference(close, length=20)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.overlap
def test_ssf3_constant_input_exact() -> None:
    """Unit DC gain (d1+d2+d3+d4 == 1): constant input reproduced exactly."""
    c = np.full(50, 42.0)
    result = ssf3_numba(c, length=10)
    assert_allclose(result, 42.0, rtol=0, atol=0)


@pytest.mark.overlap
def test_ssf3_smooths_input(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """SSF3 output is smoother (lower std) than its input."""
    close = prices_random_walk
    result = ssf3_numba(close, length=20)
    assert np.std(result[10:]) < np.std(close[10:])


@pytest.mark.overlap
def test_ssf3_seeds_first_three_samples() -> None:
    """First three outputs equal the input (no filtering yet)."""
    close = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    result = ssf3_numba(close, length=5)
    assert_allclose(result[:3], close[:3], rtol=0, atol=0)
    assert not np.isnan(result).any()


@pytest.mark.overlap
@pytest.mark.parametrize("m", [1, 2, 3])
def test_ssf3_short_inputs(m: int) -> None:
    """Inputs shorter than 4 are returned as-is (seeded from input)."""
    close = np.arange(1.0, m + 1.0)
    result = ssf3_numba(close, length=10)
    assert_allclose(result, close, rtol=0, atol=0)


@pytest.mark.overlap
def test_ssf3_empty_input() -> None:
    """Regression: empty input used to write out of bounds silently."""
    empty = np.array([], dtype=np.float64)
    result = ssf3_numba(empty, length=10)
    assert len(result) == 0


@pytest.mark.overlap
def test_ssf3_coefficients_unit_dc_gain() -> None:
    """d1 + d2 + d3 + d4 == 1 by construction (checked numerically)."""
    length = 12
    a = np.exp(-3.14159 / length)
    b = 2.0 * a * np.cos(-3.14159 * 1.732 / length)
    c = a * a
    d4 = c * c
    d3 = -c * (1.0 + b)
    d2 = b + c
    d1 = 1.0 - d2 - d3 - d4
    assert_allclose(d1 + d2 + d3 + d4, 1.0, rtol=0, atol=1e-12)


# -----------------------------------------------------------------------------
# Validation / contiguity / read-only
# -----------------------------------------------------------------------------


@pytest.mark.overlap
@pytest.mark.parametrize("length", [0, -2])
def test_ssf3_invalid_length(length: int) -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        ssf3_numba(close, length=length)


@pytest.mark.overlap
def test_ssf3_non_contiguous_input(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided input is accepted and matches the reference on that data."""
    close = prices_random_walk[::2]
    r_strided = ssf3_numba(close, length=20)
    expected = _ssf3_reference(close, length=20)
    assert_allclose(r_strided, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.overlap
def test_ssf3_ind_polars_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """ssf3_ind accepts pl.Series (read-only numpy underneath)."""
    close = prices_random_walk
    r = ssf3_ind(pl.Series(close), length=20)
    expected = ssf3_numba(close, length=20)
    assert_allclose(r, expected, rtol=0, atol=0)


# -----------------------------------------------------------------------------
# Offset / fillna
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf3_offset(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Positive offset shifts the series forward."""
    close = prices_random_walk
    r0 = ssf3_numba(close, length=20)
    r2 = ssf3_numba(close, length=20, offset=2)
    assert np.isnan(r2[:2]).all()
    assert_allclose(r2[2:], r0[:-2], rtol=0, atol=0)


@pytest.mark.overlap
def test_ssf3_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Fillna passthrough: no natural NaNs, values stay unchanged."""
    close = prices_random_walk
    r = ssf3_numba(close, length=20, fillna=-1.0)
    assert not np.isnan(r).any()
    assert_allclose(r, ssf3_numba(close, length=20), rtol=0, atol=0)


@pytest.mark.overlap
def test_ssf3_offset_with_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset NaNs are replaced by fillna."""
    close = prices_random_walk
    r = ssf3_numba(close, length=20, offset=3, fillna=0.0)
    assert (r[:3] == 0.0).all()
    assert not np.isnan(r).any()


# -----------------------------------------------------------------------------
# Polars integration
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf3_polars_default_column(df_random_walk: pl.DataFrame) -> None:
    """ssf3_polars adds an 'SSF3_{length}' column by default."""
    out = ssf3_polars(df_random_walk, length=20)
    assert "SSF3_20" in out.columns
    expected = ssf3_numba(df_random_walk["close"].to_numpy(), length=20)
    assert_allclose(out["SSF3_20"].to_numpy(), expected, rtol=0, atol=0)
    assert len(out) == len(df_random_walk)


@pytest.mark.overlap
def test_ssf3_polars_custom_column(df_random_walk: pl.DataFrame) -> None:
    """Custom output column name is honoured."""
    out = ssf3_polars(df_random_walk, length=10, output_col="MY_SSF3")
    assert "MY_SSF3" in out.columns
    assert "SSF3_10" not in out.columns


# -----------------------------------------------------------------------------
# IEEE 754 edge cases
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf3_nan_poisons_tail() -> None:
    """Recursive filter: a single NaN poisons everything after it (IIR
    feedback). Documented IIR semantics.
    """
    close = np.arange(1.0, 21.0)
    close[6] = np.nan
    result = ssf3_numba(close, length=10)
    assert np.isnan(result[6:]).all()
    assert not np.isnan(result[:6]).any()


@pytest.mark.overlap
def test_ssf3_inf_poisons_tail() -> None:
    """Inf enters the recurrence, then inf - inf turns the tail into NaN."""
    close = np.arange(1.0, 21.0)
    close[3] = np.inf
    result = ssf3_numba(close, length=10)
    assert np.isinf(result[3:5]).all()
    assert np.isnan(result[5:]).all()
    assert not np.isnan(result[:3]).any()
