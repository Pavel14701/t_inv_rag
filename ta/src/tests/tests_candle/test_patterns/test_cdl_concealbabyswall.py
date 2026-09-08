# -*- coding: utf-8 -*-
"""Tests for Concealing Baby Swallow (cdl_concealbabyswall)."""

from ._helpers import pattern_suite
from ....candle.cdl_concealbabyswall import (
    cdl_concealbabyswall,
    cdl_concealbabyswall_polars,
)

# two long blacks, then two smaller black bodies closing below
BULL = [
    (105.0, 103.0, 101.0, 103.0),
    (102.5, 100.5, 98.5, 100.5),
    (101.5, 101.8, 98.2, 98.5),
    (98.8, 99.2, 97.5, 98.2),
]

globals().update(pattern_suite(
    name='cdl_concealbabyswall',
    fn=cdl_concealbabyswall,
    polars_fn=cdl_concealbabyswall_polars,
    output_col='CDL_CONCEALBABYSWALL',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
    extra={'strict': False},
))
