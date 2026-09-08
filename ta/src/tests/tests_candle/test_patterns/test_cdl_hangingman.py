# -*- coding: utf-8 -*-
"""Tests for Hanging Man (cdl_hangingman)."""

from ._helpers import pattern_suite
from ....candle.cdl_hangingman import cdl_hangingman, cdl_hangingman_polars

# small body near the top with long lower shadow (after an uptrend)
BEAR = [
    (100.5, 101.0, 97.5, 100.0),
]

globals().update(pattern_suite(
    name='cdl_hangingman',
    fn=cdl_hangingman,
    polars_fn=cdl_hangingman_polars,
    output_col='CDL_HANGINGMAN',
    bear=BEAR,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False, 'symmetric': False},
))
