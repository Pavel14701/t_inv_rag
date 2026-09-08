# -*- coding: utf-8 -*-
"""Tests for Engulfing (cdl_engulfing)."""

from ._helpers import pattern_suite
from ....candle.cdl_engulfing import cdl_engulfing, cdl_engulfing_polars

# black body engulfed by a larger white body
BULL = [
    (102.0, 102.5, 99.5, 100.0),
    (99.8, 102.8, 99.3, 102.5),
]

# white body engulfed by a larger black body
BEAR = [
    (100.0, 100.5, 99.5, 102.0),
    (102.2, 102.6, 99.4, 99.8),
]

globals().update(pattern_suite(
    name='cdl_engulfing',
    fn=cdl_engulfing,
    polars_fn=cdl_engulfing_polars,
    output_col='CDL_ENGULFING',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
