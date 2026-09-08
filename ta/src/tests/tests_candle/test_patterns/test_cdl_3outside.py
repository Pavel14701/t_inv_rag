# -*- coding: utf-8 -*-
"""Tests for Three Outside Up/Down (cdl_3outside)."""

from ._helpers import pattern_suite
from ....candle.cdl_3outside import cdl_3outside, cdl_3outside_polars

# bearish candle, bullish engulfing, bullish continuation
BULL = [
    (102.0, 102.5, 99.5, 100.0),
    (99.8, 102.5, 99.3, 102.2),
    (102.3, 103.0, 101.8, 103.0),
]

# bullish candle, bearish engulfing, bearish continuation
BEAR = [
    (100.0, 100.5, 99.5, 102.0),
    (102.2, 102.6, 99.4, 99.8),
    (99.7, 100.2, 98.5, 98.5),
]

globals().update(pattern_suite(
    name='cdl_3outside',
    fn=cdl_3outside,
    polars_fn=cdl_3outside_polars,
    output_col='CDL_3OUTSIDE',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
