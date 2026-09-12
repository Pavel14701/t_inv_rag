# -*- coding: utf-8 -*-
"""Tests for Closing Marubozu (cdl_closingmarubozu)."""

from ._helpers import pattern_suite
from ....candle.cdl_closingmarubozu import (
    cdl_closingmarubozu,
    cdl_closingmarubozu_polars,
)

# white closing at the high
BULL = [
    (100.0, 103.0, 99.5, 103.0),
]

# black closing at the low
BEAR = [
    (103.0, 103.5, 99.5, 99.5),
]

globals().update(pattern_suite(
    name='cdl_closingmarubozu',
    fn=cdl_closingmarubozu,
    polars_fn=cdl_closingmarubozu_polars,
    output_col='CDL_CLOSINGMARUBOZU',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
