# -*- coding: utf-8 -*-
"""Tests for Rising/Falling Three Methods (cdl_risefall3methods)."""

from ._helpers import pattern_suite
from ....candle.cdl_risefall3methods import (
    cdl_risefall3methods,
    cdl_risefall3methods_polars,
)

# long white, three small blacks inside its range, white continuation
BULL = [
    (100.0, 103.4, 99.4, 103.0),
    (102.7, 103.0, 101.5, 101.7),
    (101.1, 101.5, 100.0, 100.3),
    (100.4, 100.8, 99.5, 99.7),
    (102.5, 103.6, 102.2, 103.5),
]

globals().update(pattern_suite(
    name='cdl_risefall3methods',
    fn=cdl_risefall3methods,
    polars_fn=cdl_risefall3methods_polars,
    output_col='CDL_RISEFALL3METHODS',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
