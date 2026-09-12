"""Unit tests for Upside/Downside Gap 3 Methods pattern
(Numba implementation).
"""

import numpy as np
from numpy.testing import assert_array_equal

from ....candle.cdl_xsidegap3methods import _cdl_xsidegap3methods_nb


def test_cdl_xsidegap3methods_upside():
    """Test Upside Gap 3 Methods (bullish continuation)."""
    open_ = np.array([100, 115, 118], dtype=np.float64)
    high = np.array([102, 120, 120], dtype=np.float64)
    low = np.array([98, 114, 113], dtype=np.float64)
    close = np.array([102, 118, 114], dtype=np.float64)
    expected = np.array([0, 0, 1], dtype=np.int8)
    result = _cdl_xsidegap3methods_nb(open_, high, low, close)
    assert_array_equal(result, expected)


def test_cdl_xsidegap3methods_downside():
    """Test Downside Gap 3 Methods (bearish continuation)."""
    # Candle1: black (100→98), Candle2: black with gap down (95→94),
    # Candle3: white (96→97) closing into the gap (b2_high=95, b1_low=98)
    open_ = np.array([100, 95, 96], dtype=np.float64)
    high = np.array([101, 96, 98], dtype=np.float64)
    low = np.array([99, 94, 95], dtype=np.float64)
    close = np.array([98, 94, 97], dtype=np.float64)
    expected = np.array([0, 0, 1], dtype=np.int8)
    result = _cdl_xsidegap3methods_nb(open_, high, low, close)
    assert_array_equal(result, expected)


def test_cdl_xsidegap3methods_no_pattern():
    """Test no pattern when conditions are not met."""
    open_ = np.array([100, 102, 101], dtype=np.float64)
    high = np.array([101, 103, 102], dtype=np.float64)
    low = np.array([99, 101, 100], dtype=np.float64)
    close = np.array([101, 102, 101], dtype=np.float64)
    expected = np.array([0, 0, 0], dtype=np.int8)
    result = _cdl_xsidegap3methods_nb(open_, high, low, close)
    assert_array_equal(result, expected)
