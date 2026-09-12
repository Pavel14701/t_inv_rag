# -*- coding: utf-8 -*-
"""Tests for Dragonfly Doji (cdl_dragonflydoji)."""

from ta.src.candle.cdl_dragonflydoji import (
    cdl_dragonflydoji,
    cdl_dragonflydoji_polars,
)

from ._helpers import pattern_suite


# open, high and close at the same level, long lower shadow
BULL = [
    (101.0, 101.0, 98.0, 101.0),
]

globals().update(
    pattern_suite(
        name="cdl_dragonflydoji",
        fn=cdl_dragonflydoji,
        polars_fn=cdl_dragonflydoji_polars,
        output_col="CDL_DRAGONFLYDOJI",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
        extra={"strict": False, "symmetric": False},
    )
)
