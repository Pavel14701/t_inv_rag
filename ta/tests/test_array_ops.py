"""Unit tests for Numba‑accelerated array operations (_array_ops)."""

import numpy as np
import pytest

from ta.src._array_ops import (
    _apply_offset_fillna,
    _fill_nan_policy_numba,
    _handle_nan_policy,
    _rolling_max_numba,
    _rolling_min_numba,
    replace_inf_with_nan,
)


# -----------------------------------------------------------------------------
# Tests for _apply_offset_fillna
# -----------------------------------------------------------------------------


def test_apply_offset_fillna_basic(prices_random_walk, offset, fillna):
    """Test basic shift and fillna: shape preserved, fillna applied."""
    result = _apply_offset_fillna(prices_random_walk, offset, fillna)
    assert result.shape == prices_random_walk.shape
    if offset > 0:
        np.testing.assert_array_equal(result[:offset], fillna)
        np.testing.assert_array_equal(
            result[offset:], prices_random_walk[:-offset]
        )
    elif offset < 0:
        off = -offset
        np.testing.assert_array_equal(result[-off:], fillna)
        np.testing.assert_array_equal(result[:-off], prices_random_walk[off:])
    else:
        np.testing.assert_array_equal(result, prices_random_walk)


def test_apply_offset_fillna_with_nan():
    """Test replacement of NaNs in the original array."""
    arr = np.array([1.0, np.nan, 3.0, 4.0])
    result = _apply_offset_fillna(arr, offset=0, fillna=0.0)
    expected = np.array([1.0, 0.0, 3.0, 4.0])
    np.testing.assert_array_equal(result, expected)

    result = _apply_offset_fillna(arr, offset=2, fillna=-1.0)
    expected = np.array([-1.0, -1.0, 1.0, -1.0])
    np.testing.assert_array_equal(result, expected)


def test_apply_offset_fillna_empty():
    """Empty input must return an empty array."""
    result = _apply_offset_fillna(np.array([]), offset=2, fillna=0.0)
    assert result.size == 0


def test_apply_offset_fillna_offset_gt_length():
    """Offset larger than array length => all elements become fillna."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _apply_offset_fillna(arr, offset=5, fillna=99.0)
    np.testing.assert_array_equal(result, [99.0, 99.0, 99.0])


# -----------------------------------------------------------------------------
# Tests for rolling max/min
# -----------------------------------------------------------------------------


def test_rolling_max_basic():
    """Rolling maximum on a known sequence."""
    arr = np.array([1, 3, 2, 5, 4, 2], dtype=np.float64)
    window = 3
    result = _rolling_max_numba(arr, window)
    expected = np.array([np.nan, np.nan, 3.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_almost_equal(result, expected, decimal=8)


def test_rolling_min_basic():
    """Rolling minimum on a known sequence."""
    arr = np.array([1, 3, 2, 5, 4, 2], dtype=np.float64)
    window = 3
    result = _rolling_min_numba(arr, window)
    expected = np.array([np.nan, np.nan, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(result, expected, decimal=8)


def test_rolling_max_min_with_nan():
    """NaN in the window => result is NaN for both max and min (IEEE 754)."""
    arr = np.array([1.0, np.nan, 3.0, 2.0, 5.0])
    window = 3

    max_res = _rolling_max_numba(arr, window)
    expected_max = np.array([np.nan, np.nan, np.nan, np.nan, 5.0])
    np.testing.assert_array_equal(max_res, expected_max)

    min_res = _rolling_min_numba(arr, window)
    expected_min = np.array([np.nan, np.nan, np.nan, np.nan, 2.0])
    np.testing.assert_array_equal(min_res, expected_min)


def test_rolling_max_min_empty():
    """Empty input returns empty array."""
    res_max = _rolling_max_numba(np.array([]), 3)
    res_min = _rolling_min_numba(np.array([]), 3)
    assert res_max.size == 0
    assert res_min.size == 0


def test_rolling_max_min_window_greater_than_len():
    """Window > length => all NaNs."""
    arr = np.array([1.0, 2.0, 3.0])
    res = _rolling_max_numba(arr, 5)
    np.testing.assert_array_equal(res, np.array([np.nan, np.nan, np.nan]))


def test_rolling_max_min_window_zero():
    """Window <= 0 => all NaNs."""
    arr = np.array([1.0, 2.0, 3.0])
    res = _rolling_max_numba(arr, 0)
    np.testing.assert_array_equal(res, np.array([np.nan, np.nan, np.nan]))


# -----------------------------------------------------------------------------
# Tests for NaN policy functions
# -----------------------------------------------------------------------------


def test_fill_nan_policy_ffill():
    """Forward fill: propagate last valid value."""
    arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    _fill_nan_policy_numba(arr, "ffill")
    expected = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
    np.testing.assert_array_equal(arr, expected)

    arr2 = np.array([np.nan, 2.0, np.nan, 4.0])
    _fill_nan_policy_numba(arr2, "ffill")
    expected2 = np.array([np.nan, 2.0, 2.0, 4.0])
    np.testing.assert_array_equal(arr2, expected2)


def test_fill_nan_policy_bfill():
    """Backward fill: propagate next valid value."""
    arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    _fill_nan_policy_numba(arr, "bfill")
    expected = np.array([1.0, 3.0, 3.0, 5.0, 5.0])
    np.testing.assert_array_equal(arr, expected)

    arr2 = np.array([1.0, np.nan, 3.0, np.nan])
    _fill_nan_policy_numba(arr2, "bfill")
    expected2 = np.array([1.0, 3.0, 3.0, np.nan])
    np.testing.assert_array_equal(arr2, expected2)


def test_fill_nan_policy_both():
    """'both' fills all NaNs (forward then backward), including edges."""
    arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    _fill_nan_policy_numba(arr, "both")
    expected = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
    np.testing.assert_array_equal(arr, expected)

    arr2 = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
    _fill_nan_policy_numba(arr2, "both")
    expected2 = np.array([2.0, 2.0, 2.0, 4.0, 4.0])
    np.testing.assert_array_equal(arr2, expected2)


def test_handle_nan_policy_raise():
    """'raise' policy triggers ValueError if NaN present."""
    arr = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="contains NaN"):
        _handle_nan_policy(arr, "raise", "test")


def test_handle_nan_policy_no_nan():
    """No NaNs => returns original array (no copy)."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _handle_nan_policy(arr, "raise", "test")
    assert result is arr


def test_handle_nan_policy_ignore():
    """'ignore' returns the array unchanged even with NaNs."""
    arr = np.array([1.0, np.nan, 3.0])
    result = _handle_nan_policy(arr, "ignore", "test")
    assert result is arr
    assert np.isnan(result[1])


def test_handle_nan_policy_ffill_bfill():
    """Ffill and bfill policies applied correctly via _handle_nan_policy."""
    arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    result_ff = _handle_nan_policy(arr, "ffill", "test")
    expected = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
    np.testing.assert_array_equal(result_ff, expected)

    result_bf = _handle_nan_policy(arr, "bfill", "test")
    expected_bf = np.array([1.0, 3.0, 3.0, 5.0, 5.0])
    np.testing.assert_array_equal(result_bf, expected_bf)


# -----------------------------------------------------------------------------
# Tests for replace_inf_with_nan
# -----------------------------------------------------------------------------


def test_replace_inf_with_nan():
    """Inf and -Inf are replaced with NaN in-place."""
    arr = np.array([1.0, np.inf, -np.inf, 2.0, np.nan])
    replace_inf_with_nan(arr)
    expected = np.array([1.0, np.nan, np.nan, 2.0, np.nan])
    np.testing.assert_array_equal(arr, expected)


def test_replace_inf_with_nan_empty():
    """Empty array is handled gracefully."""
    arr = np.array([])
    replace_inf_with_nan(arr)
    assert arr.size == 0


def test_replace_inf_with_nan_no_inf():
    """Array without infinities remains unchanged."""
    arr = np.array([1.0, 2.0, 3.0])
    original = arr.copy()
    replace_inf_with_nan(arr)
    np.testing.assert_array_equal(arr, original)
