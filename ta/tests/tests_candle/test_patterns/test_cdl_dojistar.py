# -*- coding: utf-8 -*-
"""Tests for Doji Star (cdl_dojistar)."""

from ._helpers import pattern_suite
from ....candle.cdl_dojistar import cdl_dojistar, cdl_dojistar_polars

# long white, then doji gapped below it (bullish doji star)
BULL = [
    (100.0, 103.0, 99.5, 102.5),
    (96.5, 99.0, 96.0, 96.2),
]

# long white, then doji gapped above it (bearish doji star)
BEAR = [
    (100.0, 100.5, 97.0, 99.5),
    (102.5, 105.5, 102.3, 102.4),
]

globals().update(pattern_suite(
    name='cdl_dojistar',
    fn=cdl_dojistar,
    polars_fn=cdl_dojistar_polars,
    output_col='CDL_DOJISTAR',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
