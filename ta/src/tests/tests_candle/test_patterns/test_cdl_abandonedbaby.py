# -*- coding: utf-8 -*-
"""Tests for Abandoned Baby (cdl_abandonedbaby)."""

from ._helpers import pattern_suite
from ....candle.cdl_abandonedbaby import (
    cdl_abandonedbaby,
    cdl_abandonedbaby_polars,
)

# bearish candle, doji with low above the first high, bullish gap-up candle
BULL = [
    (102.0, 102.5, 99.5, 100.0),
    (103.0, 103.3, 102.8, 103.0),
    (103.5, 105.8, 103.2, 105.5),
]

# bullish candle, doji with high below the first low, bearish gap-down candle
BEAR = [
    (100.0, 100.5, 99.5, 102.0),
    (98.4, 99.2, 98.3, 98.35),
    (98.0, 98.2, 96.2, 96.5),
]

globals().update(pattern_suite(
    name='cdl_abandonedbaby',
    fn=cdl_abandonedbaby,
    polars_fn=cdl_abandonedbaby_polars,
    output_col='CDL_ABANDONEDBABY',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
