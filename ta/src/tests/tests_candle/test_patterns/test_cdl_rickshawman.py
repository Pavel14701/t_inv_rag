# -*- coding: utf-8 -*-
"""Tests for Rickshaw Man (cdl_rickshawman)."""

from ._helpers import pattern_suite
from ....candle.cdl_rickshawman import cdl_rickshawman, cdl_rickshawman_polars

# doji-like body centered in a wide range with roughly equal shadows
BULL = [
    (100.0, 102.0, 98.0, 100.05),
]

globals().update(pattern_suite(
    name='cdl_rickshawman',
    fn=cdl_rickshawman,
    polars_fn=cdl_rickshawman_polars,
    output_col='CDL_RICKSHAWMAN',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
