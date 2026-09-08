# -*- coding: utf-8 -*-
"""Tests for Kicking By Length (cdl_kickingbylength)."""

from ._helpers import pattern_suite
from ....candle.cdl_kickingbylength import (
    cdl_kickingbylength,
    cdl_kickingbylength_polars,
)

# black marubozu, then a longer white marubozu gapping above its body
BULL = [
    (103.0, 103.2, 100.1, 100.1),
    (103.5, 106.8, 103.4, 106.7),
]

globals().update(pattern_suite(
    name='cdl_kickingbylength',
    fn=cdl_kickingbylength,
    polars_fn=cdl_kickingbylength_polars,
    output_col='CDL_KICKINGBYLENGTH',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
