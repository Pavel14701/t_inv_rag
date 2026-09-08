# -*- coding: utf-8 -*-
"""Tests for Spinning Top (cdl_spinningtop)."""

from ._helpers import FLAT_BODY, pattern_suite
from ....candle.cdl_spinningtop import cdl_spinningtop, cdl_spinningtop_polars

# small body with moderate shadows on both sides
BULL = [
    (100.0, 100.9, 99.1, 100.3),
]

globals().update(pattern_suite(
    name='cdl_spinningtop',
    fn=cdl_spinningtop,
    polars_fn=cdl_spinningtop_polars,
    output_col='CDL_SPINNINGTOP',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
    flat=FLAT_BODY,
))
