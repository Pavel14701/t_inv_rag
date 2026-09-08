# -*- coding: utf-8 -*-
"""Tests for Belt Hold (cdl_belthold)."""

from ._helpers import pattern_suite
from ....candle.cdl_belthold import cdl_belthold, cdl_belthold_polars

# long white candle opening at its low (bullish belt hold)
BULL = [
    (100.0, 104.0, 100.0, 103.5),
]

# long black candle opening at its high (bearish belt hold)
BEAR = [
    (104.0, 104.0, 100.5, 101.0),
]

globals().update(pattern_suite(
    name='cdl_belthold',
    fn=cdl_belthold,
    polars_fn=cdl_belthold_polars,
    output_col='CDL_BELTHOLD',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
