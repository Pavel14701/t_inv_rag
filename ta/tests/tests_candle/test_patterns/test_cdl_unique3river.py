"""Unit tests for Unique 3 River pattern (Numba implementation)."""

import numpy as np

from numpy.testing import assert_array_equal

from ta.src.candle.cdl_unique3river import _cdl_unique3river_nb


def test_cdl_unique3river_valid():
    """Test Unique 3 River (bullish reversal)."""
    open_ = np.array([100, 92, 91], dtype=np.float64)
    high = np.array([101, 93, 94], dtype=np.float64)
    low = np.array([99, 89, 90], dtype=np.float64)
    close = np.array([90, 91, 93], dtype=np.float64)
    expected = np.array([0, 0, 1], dtype=np.int8)
    result = _cdl_unique3river_nb(open_, high, low, close)
    assert_array_equal(result, expected)


def test_cdl_unique3river_invalid():
    """Test that missing any condition fails detection."""
    # Third candle has a large body (not small)
    open_ = np.array([100, 92, 91], dtype=np.float64)
    high = np.array([101, 93, 95], dtype=np.float64)
    low = np.array([99, 89, 89], dtype=np.float64)
    # body = 4 (> 0.6 * range = 3.6)
    close = np.array([90, 91, 95], dtype=np.float64)
    expected = np.array([0, 0, 0], dtype=np.int8)
    result = _cdl_unique3river_nb(open_, high, low, close)
    assert_array_equal(result, expected)
