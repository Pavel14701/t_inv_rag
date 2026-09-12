"""Unit tests for Upside Gap Two Crows pattern (Numba implementation)."""

import numpy as np

from numpy.testing import assert_array_equal

from ta.src.candle.cdl_upsidegap2crows import _cdl_upsidegap2crows_nb


def test_cdl_upsidegap2crows_valid():
    """Test Upside Gap Two Crows (bearish reversal)."""
    open_ = np.array([100, 115, 118], dtype=np.float64)
    high = np.array([102, 116, 119], dtype=np.float64)
    low = np.array([98, 114, 110], dtype=np.float64)
    close = np.array([110, 112, 111], dtype=np.float64)
    expected = np.array([0, 0, 1], dtype=np.int8)
    result = _cdl_upsidegap2crows_nb(open_, high, low, close)
    assert_array_equal(result, expected)


def test_cdl_upsidegap2crows_invalid():
    """Test that missing any condition fails detection."""
    open_ = np.array([100, 115, 118], dtype=np.float64)
    high = np.array([102, 116, 119], dtype=np.float64)
    low = np.array([98, 114, 110], dtype=np.float64)
    close = np.array([110, 112, 113], dtype=np.float64)  # close above c2
    expected = np.array([0, 0, 0], dtype=np.int8)
    result = _cdl_upsidegap2crows_nb(open_, high, low, close)
    assert_array_equal(result, expected)
