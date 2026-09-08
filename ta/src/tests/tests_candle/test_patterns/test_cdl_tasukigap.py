# -*- coding: utf-8 -*-
"""Tests for Tasuki Gap (cdl_tasukigap)."""

from ._helpers import pattern_suite
from ....candle.cdl_tasukigap import cdl_tasukigap, cdl_tasukigap_polars

# two whites with an upward gap, then a black pulling back into the gap
BULL = [
    (100.0, 100.5, 99.5, 102.0),
    (102.5, 103.5, 101.0, 103.0),
    (102.8, 103.0, 100.6, 100.8),
]

globals().update(pattern_suite(
    name='cdl_tasukigap',
    fn=cdl_tasukigap,
    polars_fn=cdl_tasukigap_polars,
    output_col='CDL_TASUKIGAP',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
