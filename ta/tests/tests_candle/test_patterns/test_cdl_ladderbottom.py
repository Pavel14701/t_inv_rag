# -*- coding: utf-8 -*-
"""Tests for Ladder Bottom (cdl_ladderbottom)."""

from ta.src.candle.cdl_ladderbottom import (
    cdl_ladderbottom,
    cdl_ladderbottom_polars,
)

from ._helpers import pattern_suite


# two blacks, long lower shadow on the last, then a strong white
BULL = [
    (104.0, 104.3, 102.8, 103.0),
    (102.8, 103.0, 101.3, 101.5),
    (101.5, 101.8, 100.0, 100.2),
    (100.0, 100.2, 94.0, 99.0),
    (99.0, 101.5, 98.5, 101.0),
]

globals().update(
    pattern_suite(
        name="cdl_ladderbottom",
        fn=cdl_ladderbottom,
        polars_fn=cdl_ladderbottom_polars,
        output_col="CDL_LADDERBOTTOM",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
