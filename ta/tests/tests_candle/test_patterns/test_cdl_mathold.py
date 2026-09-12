# -*- coding: utf-8 -*-
"""Tests for Mat Hold (cdl_mathold)."""

from ._helpers import pattern_suite
from ....candle.cdl_mathold import cdl_mathold, cdl_mathold_polars

# long white, small-bodied pullback within its range, white continuation
BULL = [
    (100.0, 103.4, 99.4, 103.0),
    (102.9, 103.0, 101.2, 101.5),
    (101.4, 101.7, 100.1, 100.4),
    (100.5, 100.8, 100.0, 100.3),
    (102.9, 103.8, 102.5, 103.5),
]

globals().update(pattern_suite(
    name='cdl_mathold',
    fn=cdl_mathold,
    polars_fn=cdl_mathold_polars,
    output_col='CDL_MATHOLD',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
