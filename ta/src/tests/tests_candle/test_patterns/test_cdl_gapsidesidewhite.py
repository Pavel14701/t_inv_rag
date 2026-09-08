# -*- coding: utf-8 -*-
"""Tests for Gap Side-by-Side White Lines (cdl_gapsidesidewhite)."""

from ._helpers import pattern_suite
from ....candle.cdl_gapsidesidewhite import (
    cdl_gapsidesidewhite,
    cdl_gapsidesidewhite_polars,
)

# white, then two whites gapping up with similar bodies
BULL = [
    (100.0, 100.5, 99.5, 102.0),
    (102.2, 103.2, 101.0, 103.0),
    (102.5, 103.7, 102.0, 103.2),
]

# black, then two whites gapping down with similar bodies
BEAR = [
    (102.0, 102.5, 99.5, 100.0),
    (96.8, 99.2, 96.0, 98.5),
    (96.5, 98.8, 96.1, 98.4),
]

globals().update(pattern_suite(
    name='cdl_gapsidesidewhite',
    fn=cdl_gapsidesidewhite,
    polars_fn=cdl_gapsidesidewhite_polars,
    output_col='CDL_GAPSIDESIDEWHITE',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
