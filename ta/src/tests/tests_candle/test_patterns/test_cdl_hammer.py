# -*- coding: utf-8 -*-
"""Tests for Hammer (cdl_hammer)."""

from ._helpers import pattern_suite
from ....candle.cdl_hammer import cdl_hammer, cdl_hammer_polars

# small body near the top, long lower shadow
BULL = [
    (100.5, 101.0, 97.5, 100.0),
]

globals().update(pattern_suite(
    name='cdl_hammer',
    fn=cdl_hammer,
    polars_fn=cdl_hammer_polars,
    output_col='CDL_HAMMER',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
    extra={'strict': False, 'symmetric': False},
))
