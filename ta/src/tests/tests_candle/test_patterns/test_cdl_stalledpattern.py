# -*- coding: utf-8 -*-
"""Tests for Stalled Pattern (cdl_stalledpattern)."""

from ._helpers import pattern_suite
from ....candle.cdl_stalledpattern import (
    cdl_stalledpattern,
    cdl_stalledpattern_polars,
)

# two long whites, then a small white with a long upper shadow stalling
BULL = [
    (100.0, 100.4, 99.4, 103.0),
    (103.0, 103.8, 102.9, 103.6),
    (103.6, 104.05, 103.55, 103.75),
]

globals().update(pattern_suite(
    name='cdl_stalledpattern',
    fn=cdl_stalledpattern,
    polars_fn=cdl_stalledpattern_polars,
    output_col='CDL_STALLEDPATTERN',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
