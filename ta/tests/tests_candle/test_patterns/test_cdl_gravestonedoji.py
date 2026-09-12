# -*- coding: utf-8 -*-
"""Tests for Gravestone Doji (cdl_gravestonedoji)."""

from ta.src.candle.cdl_gravestonedoji import (
    cdl_gravestonedoji,
    cdl_gravestonedoji_polars,
)

from ._helpers import pattern_suite


# open, low and close at the same level, long upper shadow
BEAR = [
    (98.0, 101.0, 98.0, 98.0),
]

globals().update(
    pattern_suite(
        name="cdl_gravestonedoji",
        fn=cdl_gravestonedoji,
        polars_fn=cdl_gravestonedoji_polars,
        output_col="CDL_GRAVESTONEDOJI",
        bear=BEAR,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
        extra={"strict": False, "symmetric": False},
    )
)
