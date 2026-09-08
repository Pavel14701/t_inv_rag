# -*- coding: utf-8 -*-
"""Tests for Three-Line Strike (cdl_3linestrike)."""

from ._helpers import pattern_suite
from ....candle.cdl_3linestrike import (
    cdl_3linestrike,
    cdl_3linestrike_polars,
)

# three rising whites, then a black strike closing below the first white
BULL = [
    (100.0, 100.5, 99.5, 102.0),
    (102.5, 103.0, 102.0, 104.0),
    (105.0, 105.5, 104.5, 107.0),
    (108.0, 108.5, 100.5, 101.0),
]

# three falling blacks, then a white strike closing above the first black
BEAR = [
    (102.0, 102.5, 99.5, 100.0),
    (99.8, 100.2, 96.8, 97.5),
    (97.0, 97.4, 94.0, 95.0),
    (94.0, 101.5, 93.5, 101.0),
]

globals().update(pattern_suite(
    name='cdl_3linestrike',
    fn=cdl_3linestrike,
    polars_fn=cdl_3linestrike_polars,
    output_col='CDL_3LINESTRIKE',
    bull=BULL,
    bear=BEAR,
    bull_value=1.0,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
