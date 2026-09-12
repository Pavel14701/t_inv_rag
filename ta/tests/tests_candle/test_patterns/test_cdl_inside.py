# -*- coding: utf-8 -*-
"""Tests for Inside bar pattern (cdl_inside)."""

from ._helpers import pattern_suite
from ....candle.cdl_inside import cdl_inside, cdl_inside_polars

# second candle entirely within the first candle's range
BULL = [
    (100.0, 103.0, 97.0, 101.0),
    (101.0, 102.0, 98.0, 100.5),
]

globals().update(pattern_suite(
    name='cdl_inside',
    fn=cdl_inside,
    polars_fn=cdl_inside_polars,
    output_col='CDL_INSIDE',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
