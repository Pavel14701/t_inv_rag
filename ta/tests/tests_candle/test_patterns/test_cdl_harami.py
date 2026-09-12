# -*- coding: utf-8 -*-
"""Tests for Harami (cdl_harami)."""

from ta.src.candle.cdl_harami import cdl_harami, cdl_harami_polars

from ._helpers import pattern_suite


# large black body, small white body inside it
BULL = [
    (103.0, 103.5, 99.5, 100.0),
    (100.2, 102.3, 100.0, 102.0),
]

# large white body, small black body inside it
BEAR = [
    (100.0, 100.5, 99.5, 103.0),
    (102.8, 103.0, 99.8, 100.2),
]

globals().update(
    pattern_suite(
        name="cdl_harami",
        fn=cdl_harami,
        polars_fn=cdl_harami_polars,
        output_col="CDL_HARAMI",
        bull=BULL,
        bear=BEAR,
        bull_value=1.0,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
        extra={"strict": False},
    )
)
