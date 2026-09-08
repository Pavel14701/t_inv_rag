# -*- coding: utf-8 -*-
"""Tests for Stick Sandwich (cdl_sticksandwich)."""

from ._helpers import pattern_suite
from ....candle.cdl_sticksandwich import (
    cdl_sticksandwich,
    cdl_sticksandwich_polars,
)

# long black, white inside its body, black closing exactly at the first close
BULL = [
    (103.0, 103.5, 99.5, 100.0),
    (101.0, 102.2, 100.5, 101.8),
    (103.1, 103.6, 99.6, 100.0),
]

globals().update(pattern_suite(
    name='cdl_sticksandwich',
    fn=cdl_sticksandwich,
    polars_fn=cdl_sticksandwich_polars,
    output_col='CDL_STICKSANDWICH',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
