"""Unit tests for Tristar pattern (Numba implementation)."""

import numpy as np
from numpy.testing import assert_array_equal

from ....candle.cdl_tristar import _cdl_tristar_nb


def test_cdl_tristar_valid():
    """Test Tristar (bullish)."""
    open_ = np.array([100, 105, 110], dtype=np.float64)
    high = np.array([101, 106, 111], dtype=np.float64)
    low = np.array([99, 104, 109], dtype=np.float64)
    close = np.array([100, 105, 110], dtype=np.float64)
    expected = np.array([0, 0, 1], dtype=np.int8)

    result = _cdl_tristar_nb(open_, high, low, close)
    assert_array_equal(result, expected)


def test_cdl_tristar_no_gap():
    """Test no pattern when gaps are missing."""
    open_ = np.array([100, 100, 100], dtype=np.float64)
    high = np.array([101, 101, 101], dtype=np.float64)
    low = np.array([99, 99, 99], dtype=np.float64)
    close = np.array([100, 100, 100], dtype=np.float64)
    expected = np.array([0, 0, 0], dtype=np.int8)

    result = _cdl_tristar_nb(open_, high, low, close)
    assert_array_equal(result, expected)
