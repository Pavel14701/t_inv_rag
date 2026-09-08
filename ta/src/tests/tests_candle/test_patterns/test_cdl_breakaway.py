# -*- coding: utf-8 -*-
"""Tests for Breakaway (cdl_breakaway)."""

from ._helpers import pattern_suite
from ....candle.cdl_breakaway import cdl_breakaway, cdl_breakaway_polars

# four blacks in a downtrend, then a white closing into the gap
BULL = [
    (105.0, 105.5, 103.0, 103.2),
    (102.5, 103.0, 100.0, 100.5),
    (100.0, 100.5, 97.5, 98.0),
    (97.5, 98.0, 95.0, 95.5),
    (95.2, 104.0, 95.0, 103.8),
]

# four whites in an uptrend, then a black closing into the gap
BEAR = [
    (95.0, 97.5, 94.5, 96.8),
    (98.0, 100.5, 97.5, 100.0),
    (100.5, 103.0, 100.0, 102.5),
    (103.0, 105.5, 102.5, 105.0),
    (105.8, 106.0, 96.0, 96.5),
]

globals().update(pattern_suite(
    name='cdl_breakaway',
    fn=cdl_breakaway,
    polars_fn=cdl_breakaway_polars,
    output_col='CDL_BREAKAWAY',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
