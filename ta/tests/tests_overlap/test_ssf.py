# -*- coding: utf-8 -*-
"""Unit tests for Ehlers Super Smoother Filter (SSF) module.

Tests cover:
- Regression: the Ehlers variant uses pi (radians), matching Everget's
  recurrence (previously `cos(180 * ratio)` was evaluated in radians)
- Both kernels against a pure Python reference
- everget=True/False consistency for the same pi
- Seed semantics (first `length`-independent samples) and short inputs
- Validation of `length`
- offset and fillna behaviour
- ssf_ind with Polars Series / ssf_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.overlap.ssf import ssf_ind, ssf_numba, ssf_polars


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------


def _ssf_reference(
    close: npt.NDArray[np.float64],
    length: int,
    pi: float,
    sqrt2: float,
) -> npt.NDArray[np.float64]:
    """Pure Python 2-pole Super Smoother (Everget form)."""
    m = len(close)
    out = np.empty(m, dtype=np.float64)
    arg = pi * sqrt2 / length
    a = np.exp(-arg)
    b = 2.0 * a * np.cos(arg)
    out[0] = close[0]
    if m > 1:
        out[1] = close[1]
    for i in range(2, m):
        out[i] = (
            0.5 * (a * a - b + 1.0) * (close[i] + close[i - 1])
            + b * out[i - 1]
            - a * a * out[i - 2]
        )
    return out


# -----------------------------------------------------------------------------
# Regression / correctness
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf_ehlers_matches_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Regression: the Ehlers variant now uses pi*ratio (radians) instead of
    the broken cos(180*ratio) (evaluated in radians => wrong by 180/pi).
    """
    close = prices_random_walk
    result = ssf_numba(close, length=20, everget=False)
    expected = _ssf_reference(close, 20, pi=3.14159, sqrt2=1.414)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.overlap
def test_ssf_everget_matches_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Everget variant matches its pure Python recurrence."""
    close = prices_random_walk
    result = ssf_numba(close, length=20, everget=True)
    expected = _ssf_reference(close, 20, pi=3.14159, sqrt2=1.414)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.overlap
def test_ssf_variants_agree_for_same_pi(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Regression: with the degrees-bug fixed, both variants implement the
    identical recurrence and must agree to floating-point precision.
    """
    close = prices_random_walk
    r_ehlers = ssf_numba(close, length=20, everget=False)
    r_everget = ssf_numba(close, length=20, everget=True)
    assert_allclose(r_ehlers, r_everget, rtol=0, atol=1e-12)


@pytest.mark.overlap
def test_ssf_pi_precision_changes_result(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Using a different pi (np.pi) changes the output only slightly —
    the variants differ historically only in pi precision.
    """
    close = prices_random_walk
    r_default = ssf_numba(close, length=20)
    r_npp = ssf_numba(close, length=20, pi=np.pi)
    diff = np.max(np.abs(r_default[2:] - r_npp[2:]))
    assert diff > 0.0
    assert diff < 1e-3


# -----------------------------------------------------------------------------
# Smoothing properties
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf_smooths_input(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """SSF output is smoother (lower std) than its input."""
    close = prices_random_walk
    result = ssf_numba(close, length=20)
    assert np.std(result[10:]) < np.std(close[10:])


@pytest.mark.overlap
def test_ssf_constant_input() -> None:
    """Constant input is reproduced (unit DC gain of the filter)."""
    c = np.full(50, 42.0)
    result = ssf_numba(c, length=10)
    assert_allclose(result, 42.0, rtol=1e-9, atol=1e-9)


@pytest.mark.overlap
def test_ssf_finite_output(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """No warm-up NaNs: the filter seeds directly from the input."""
    close = prices_random_walk
    result = ssf_numba(close, length=20)
    assert not np.isnan(result).any()


@pytest.mark.overlap
@pytest.mark.parametrize("everget", [False, True])
def test_ssf_short_inputs_do_not_corrupt(everget: bool) -> None:
    """Regression: with bounds checking disabled, writing out[1] on a
    1-element array silently corrupted memory; short inputs now seed safely.
    """
    one = np.array([5.0])
    assert_allclose(ssf_numba(one, length=10, everget=everget), [5.0])

    two = np.array([5.0, 7.0])
    assert_allclose(ssf_numba(two, length=10, everget=everget), two)


# -----------------------------------------------------------------------------
# Validation / contiguity / read-only
# -----------------------------------------------------------------------------


@pytest.mark.overlap
@pytest.mark.parametrize("length", [0, -5])
def test_ssf_invalid_length(length: int) -> None:
    """Length below 1 raises ValueError."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        ssf_numba(close, length=length)


@pytest.mark.overlap
def test_ssf_non_contiguous_input(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided input is accepted and matches the reference on that data."""
    close = prices_random_walk[::2]
    r_strided = ssf_numba(close, length=20)
    expected = _ssf_reference(close, 20, pi=3.14159, sqrt2=1.414)
    assert_allclose(r_strided, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.overlap
def test_ssf_ind_polars_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """ssf_ind accepts pl.Series (read-only numpy underneath)."""
    close = prices_random_walk
    r = ssf_ind(pl.Series(close), length=20)
    expected = ssf_numba(close, length=20)
    assert_allclose(r, expected, rtol=0, atol=0)


# -----------------------------------------------------------------------------
# Offset / fillna
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf_offset(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Positive offset shifts the series forward."""
    close = prices_random_walk
    r0 = ssf_numba(close, length=20)
    r2 = ssf_numba(close, length=20, offset=2)
    assert np.isnan(r2[:2]).all()
    assert_allclose(r2[2:], r0[:-2], rtol=0, atol=0)


@pytest.mark.overlap
def test_ssf_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Fillna passthrough: no natural NaNs, values stay unchanged."""
    close = prices_random_walk
    r = ssf_numba(close, length=20, fillna=-1.0)
    assert not np.isnan(r).any()
    assert_allclose(r, ssf_numba(close, length=20), rtol=0, atol=0)


# -----------------------------------------------------------------------------
# Polars integration
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_ssf_polars_default_column(df_random_walk: pl.DataFrame) -> None:
    """ssf_polars adds an 'SSF_{length}' column by default."""
    out = ssf_polars(df_random_walk, length=20)
    assert "SSF_20" in out.columns
    expected = ssf_numba(df_random_walk["close"].to_numpy(), length=20)
    assert_allclose(out["SSF_20"].to_numpy(), expected, rtol=0, atol=0)
    assert len(out) == len(df_random_walk)


@pytest.mark.overlap
def test_ssf_polars_everget_column(df_random_walk: pl.DataFrame) -> None:
    """everget=True produces the 'SSFe_{length}' column."""
    out = ssf_polars(df_random_walk, length=10, everget=True)
    assert "SSFe_10" in out.columns


@pytest.mark.overlap
def test_ssf_polars_custom_column(df_random_walk: pl.DataFrame) -> None:
    """Custom output column name is honoured."""
    out = ssf_polars(df_random_walk, length=10, output_col="MY_SSF")
    assert "MY_SSF" in out.columns
    assert "SSF_10" not in out.columns


# -----------------------------------------------------------------------------
# IEEE 754 edge cases
# -----------------------------------------------------------------------------


@pytest.mark.overlap
@pytest.mark.parametrize("everget", [False, True])
def test_ssf_nan_poisons_tail(everget: bool) -> None:
    """Recursive filter: a single NaN poisons everything after it (IIR
    feedback). Documented IIR semantics.
    """
    close = np.arange(1.0, 21.0)
    close[5] = np.nan
    result = ssf_numba(close, length=10, everget=everget)
    assert np.isnan(result[5:]).all()
    assert not np.isnan(result[:5]).any()


@pytest.mark.overlap
@pytest.mark.parametrize("everget", [False, True])
def test_ssf_inf_poisons_tail(everget: bool) -> None:
    """Inf enters the recurrence, then inf - inf turns the tail into NaN."""
    close = np.arange(1.0, 21.0)
    close[4] = np.inf
    result = ssf_numba(close, length=10, everget=everget)
    assert np.isinf(result[4:6]).all()
    assert np.isnan(result[6:]).all()
    assert not np.isnan(result[:4]).any()


@pytest.mark.overlap
@pytest.mark.parametrize("everget", [False, True])
def test_ssf_empty_input(everget: bool) -> None:
    """Regression: empty input used to write out of bounds silently
    (numba bounds checking is off).
    """
    empty = np.array([], dtype=np.float64)
    result = ssf_numba(empty, length=10, everget=everget)
    assert len(result) == 0
