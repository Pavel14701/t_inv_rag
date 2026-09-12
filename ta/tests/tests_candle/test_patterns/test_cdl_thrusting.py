# -*- coding: utf-8 -*-
"""Tests for Thrusting (cdl_thrusting)."""

from ta.src.candle.cdl_thrusting import cdl_thrusting, cdl_thrusting_polars

from ._helpers import pattern_suite


# long black, gap-down white closing back into the body below the midpoint
BULL = [
    (103.0, 103.5, 99.5, 100.0),
    (99.2, 101.0, 98.9, 100.8),
]

globals().update(
    pattern_suite(
        name="cdl_thrusting",
        fn=cdl_thrusting,
        polars_fn=cdl_thrusting_polars,
        output_col="CDL_THRUSTING",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
