# -*- coding: utf-8 -*-
"""Tests for Three Stars In The South (cdl_3starsinsouth)."""

from ta.src.candle.cdl_3starsinsouth import (
    cdl_3starsinsouth,
    cdl_3starsinsouth_polars,
)

from ._helpers import pattern_suite


# three black candles with falling lows; the third closes above the second
BULL = [
    (102.0, 102.5, 99.0, 100.0),
    (101.5, 102.0, 98.5, 99.5),
    (101.0, 101.4, 98.0, 100.0),
]

globals().update(
    pattern_suite(
        name="cdl_3starsinsouth",
        fn=cdl_3starsinsouth,
        polars_fn=cdl_3starsinsouth_polars,
        output_col="CDL_3STARSINSOUTH",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
        extra={"strict": False, "symmetric": False},
    )
)
