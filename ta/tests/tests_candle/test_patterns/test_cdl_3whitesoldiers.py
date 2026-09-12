# -*- coding: utf-8 -*-
"""Tests for Three White Soldiers (cdl_3whitesoldiers)."""

from ._helpers import pattern_suite
from ....candle.cdl_3whitesoldiers import (
    cdl_3whitesoldiers,
    cdl_3whitesoldiers_polars,
)

# three rising whites with rising opens and closes
BULL = [
    (100.0, 100.6, 99.5, 102.0),
    (102.2, 102.8, 101.7, 104.0),
    (104.2, 104.8, 103.7, 106.0),
]

globals().update(pattern_suite(
    name='cdl_3whitesoldiers',
    fn=cdl_3whitesoldiers,
    polars_fn=cdl_3whitesoldiers_polars,
    output_col='CDL_3WHITESOLDIERS',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
    extra={'strict': False, 'symmetric': False},
))
