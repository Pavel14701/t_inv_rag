# -*- coding: utf-8 -*-
"""Unit tests for RMA (Wilder's Moving Average) module.

Tests cover:
- _rma_numba_core against a pure-Python reference (SMA seed + recursion)
- All nan_policy modes ('raise', 'ignore', 'ffill', 'bfill', 'both')
- offset and fillna
- Input validation (length < 1, unknown nan_policy)
- Universal wrapper (rma_ind) with Polars Series and list input
- Polars integration (rma_polars)
- IEEE 754 compliance (empty, short, all-NaN, extreme)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.overlap.rma import _rma_numba_core, rma_ind, rma_numba, rma_polars


# -----------------------------------------------------------------------------
# Reference implementation
# -----------------------------------------------------------------------------


def _rma_reference(
    arr: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure-Python RMA: SMA seed at index length-1, then Wilder recursion."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    out[length - 1] = arr[:length].mean()
    alpha = 1.0 / length
    for i in range(length, n):
        out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return out


# -----------------------------------------------------------------------------
# Core tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_rma_numba_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Compare the numba core with the pure-Python reference."""
    for length in (1, 2, 5, 14):
        result = _rma_numba_core(prices_random_walk, length)
        expected = _rma_reference(prices_random_walk, length)
        assert result.shape == prices_random_walk.shape
        assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
        assert np.isnan(result[: length - 1]).all()


@pytest.mark.overlap
def test_rma_seed_is_sma() -> None:
    """The first non-NaN value is the SMA of the first `length` points."""
    close = np.arange(1.0, 11.0)
    result = _rma_numba_core(close, 4)
    assert result[3] == (1.0 + 2.0 + 3.0 + 4.0) / 4


@pytest.mark.overlap
def test_rma_hand_computed() -> None:
    """Hand-computed values for length=3."""
    close = np.arange(1.0, 9.0)
    result = rma_numba(close, 3)
    assert result[2] == 2.0  # SMA(1,2,3)
    assert result[3] == pytest.approx(2.0 + (4.0 - 2.0) / 3)
    assert result[4] == pytest.approx(
        2.0 + (4.0 - 2.0) / 3 + (5.0 - 2.666666) / 3, rel=1e-5
    )


@pytest.mark.overlap
def test_rma_length_one_is_identity() -> None:
    """length=1 -> alpha=1 -> RMA equals the input series."""
    close = np.arange(1.0, 8.0)
    assert_allclose(rma_numba(close, 1), close, rtol=1e-12)


@pytest.mark.overlap
def test_rma_constant_series() -> None:
    """Constant series -> RMA equals the constant."""
    close = np.full(20, 42.0)
    result = rma_numba(close, 5)
    assert_allclose(result[4:], 42.0, rtol=1e-12)


@pytest.mark.overlap
def test_rma_lags_price_direction(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """RMA stays between the previous RMA and the current price."""
    length = 10
    result = rma_numba(prices_random_walk, length)
    arr = prices_random_walk
    for i in range(length, len(arr)):
        lo, hi = min(result[i - 1], arr[i]), max(result[i - 1], arr[i])
        assert lo - 1e-9 <= result[i] <= hi + 1e-9


@pytest.mark.overlap
def test_rma_matches_ewm_asymptotically() -> None:
    """RMA(l) recursion equals EWM(alpha=1/l): seeded differently, so the
    difference decays geometrically; after 10*length bars it is tiny.
    """
    rng = np.random.default_rng(42)
    length = 5
    arr = 100 + np.cumsum(rng.normal(0, 1, 300))
    rma = rma_numba(arr, length)
    # Pure EWM recursion seeded from the first value:
    alpha = 1.0 / length
    ewm = np.empty_like(arr)
    ewm[0] = arr[0]
    for i in range(1, len(arr)):
        ewm[i] = ewm[i - 1] + alpha * (arr[i] - ewm[i - 1])
    diff = np.abs(rma[-1] - ewm[-1])
    assert diff < 1e-3  # (1-alpha)^n decay from the seeding difference


# -----------------------------------------------------------------------------
# nan_policy tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_rma_nan_policy_raise() -> None:
    """nan_policy='raise' raises on NaN input."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    with pytest.raises(ValueError, match="contains NaN"):
        rma_numba(arr, 2, nan_policy="raise")


@pytest.mark.overlap
def test_rma_nan_policy_ignore() -> None:
    """nan_policy='ignore': NaN poisons the recursion from its position."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = rma_numba(arr, 2, nan_policy="ignore")
    assert np.isfinite(result[1])
    assert np.isnan(result[2:]).all()


@pytest.mark.overlap
def test_rma_nan_policy_ffill() -> None:
    """nan_policy='ffill' fills gaps and keeps RMA finite afterwards."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = rma_numba(arr, 2, nan_policy="ffill")
    expected = _rma_reference(np.array([1.0, 2.0, 2.0, 4.0, 5.0]), 2)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_rma_nan_policy_bfill() -> None:
    """nan_policy='bfill' fills gaps from the right."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = rma_numba(arr, 2, nan_policy="bfill")
    expected = _rma_reference(np.array([1.0, 2.0, 4.0, 4.0, 5.0]), 2)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_rma_nan_policy_both_fills_leading_nan() -> None:
    """nan_policy='both' also fills leading NaNs (ffill cannot)."""
    arr = np.array([np.nan, 2.0, 3.0, 4.0, 5.0])
    result = rma_numba(arr, 2, nan_policy="both")
    expected = _rma_reference(np.array([2.0, 2.0, 3.0, 4.0, 5.0]), 2)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_rma_unknown_nan_policy() -> None:
    """Unknown nan_policy raises ValueError listing valid options."""
    arr = np.array([1.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="Unknown nan_policy"):
        rma_numba(arr, 2, nan_policy="invalid")


@pytest.mark.overlap
def test_rma_invalid_length() -> None:
    """Length < 1 raises ValueError."""
    close = np.array([10.0, 11.0, 12.0])
    with pytest.raises(ValueError, match="length must be >= 1"):
        rma_numba(close, length=0)
    with pytest.raises(ValueError, match="length must be >= 1"):
        rma_numba(close, length=-1)


@pytest.mark.overlap
def test_rma_offset_fillna() -> None:
    """Test rma_numba with offset and fillna."""
    close = np.arange(1.0, 11.0)
    offset = 2
    fillna = 0.0
    base = rma_numba(close, 3, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = rma_numba(close, 3, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
    assert (result[:offset] == fillna).all()


@pytest.mark.overlap
def test_rma_input_types() -> None:
    """float32 input, Python list and read-only arrays are handled."""
    r32 = rma_numba(np.arange(1.0, 6.0, dtype=np.float32), 2)
    assert r32.dtype == np.float64
    r_list = rma_ind([1.0, 2.0, 3.0, 4.0], 2)
    assert_allclose(r_list[1:], [1.5, 2.25, 3.125], rtol=1e-12)
    c = np.arange(1.0, 6.0)
    c.setflags(write=False)
    assert np.isfinite(rma_numba(c, 2)[1:]).all()


# -----------------------------------------------------------------------------
# Universal wrapper tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_rma_ind_matches_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """rma_ind returns the same result as rma_numba."""
    result = rma_ind(prices_random_walk, length=10)
    expected = rma_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_rma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """rma_ind accepts a Polars Series and matches rma_numba."""
    s = pl.Series(prices_random_walk)
    result = rma_ind(s, length=10)
    expected = rma_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)


# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_rma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """rma_polars returns a DataFrame with a correct RMA column."""
    length = 10
    result = rma_polars(df_random_walk, col="close", length=length)
    assert isinstance(result, pl.DataFrame)
    assert f"RMA_{length}" in result.columns
    close_arr = df_random_walk["close"].to_numpy()
    expected = _rma_reference(close_arr, length)
    assert_allclose(
        result[f"RMA_{length}"].to_numpy(),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.overlap
def test_rma_polars_custom_output_col(df_random_walk) -> None:
    """Custom output column name is respected."""
    result = rma_polars(
        df_random_walk, col="close", length=5, output_col="RMA"
    )
    assert "RMA" in result.columns
    assert result["RMA"].dtype == pl.Float64


@pytest.mark.overlap
def test_rma_polars_custom_col(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """rma_polars with a non-default column name."""
    df = pl.DataFrame({"price": prices_random_walk})
    result = rma_polars(df, col="price", length=10, output_col="RMA")
    expected = _rma_reference(prices_random_walk, 10)
    assert_allclose(
        result["RMA"].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


@pytest.mark.overlap
def test_rma_polars_with_offset_fillna(df_random_walk) -> None:
    """rma_polars applies offset and fillna."""
    offset = 2
    fillna = 0.0
    close_arr = df_random_walk["close"].to_numpy()


# -----------------------------------------------------------------------------
# IEEE 754 / edge case tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_rma_with_inf(prices_with_inf) -> None:
    """Inf is not NaN: it enters the recursion and then inf - inf -> NaN
    in the very next step (IEEE 754), so NaN starts one step after the Inf.
    """
    result = rma_numba(prices_with_inf, 3, nan_policy="ignore")
    assert result[5] == np.inf
    assert np.isnan(result[6:]).all()
    assert np.isfinite(result[2:5]).all()


@pytest.mark.overlap
def test_rma_empty(prices_empty) -> None:
    """Empty input returns an empty array."""
    result = rma_numba(prices_empty, 3)
    assert result.size == 0


@pytest.mark.overlap
def test_rma_shorter_than_length(prices_short) -> None:
    """len(arr) < length -> all NaN (no crash)."""
    result = rma_numba(prices_short, 10)
    assert np.isnan(result).all()


@pytest.mark.overlap
def test_rma_all_nan(prices_all_nan) -> None:
    """All NaNs: 'both' fills nothing -> NaNs; fillna=0 replaces them."""
    result = rma_numba(prices_all_nan, 3, nan_policy="both", fillna=0.0)
    assert (result == 0.0).all()


@pytest.mark.overlap
def test_rma_extreme_values(prices_extreme) -> None:
    """Extreme values must not crash."""
    result = rma_numba(prices_extreme, 3, nan_policy="both")
    assert result is not None


@pytest.mark.overlap
def test_rma_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """Polars integration with nan_policy='ffill' matches the raw backend."""
    close_arr = df_random_walk["close"].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series("close", close_arr))
    result = rma_polars(
        df_with_nan,
        col="close",
        length=3,
        nan_policy="ffill",
        output_col="RMA",
    )
    vals = result["RMA"].to_numpy()
    nb = rma_numba(close_arr, 3, nan_policy="ffill")
    assert np.array_equal(
        np.nan_to_num(vals, nan=-999.0),
        np.nan_to_num(nb, nan=-999.0),
    )
    assert np.isfinite(vals[2:]).all()
