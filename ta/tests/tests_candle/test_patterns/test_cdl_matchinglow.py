# -*- coding: utf-8 -*-
"""Tests for Matching Low (cdl_matchinglow)."""

from ._helpers import pattern_suite
from ....candle.cdl_matchinglow import cdl_matchinglow, cdl_matchinglow_polars

# two black candles with identical lows, the second closing higher
BULL = [
    (101.0, 101.5, 98.0, 98.5),
    (98.7, 98.9, 98.0, 98.6),
]

globals().update(pattern_suite(
    name='cdl_matchinglow',
    fn=cdl_matchinglow,
    polars_fn=cdl_matchinglow_polars,
    output_col='CDL_MATCHINGLOW',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
