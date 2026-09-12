# -*- coding: utf-8 -*-
"""Tests for High-Wave (cdl_highwave)."""

from ta.src.candle.cdl_highwave import cdl_highwave, cdl_highwave_polars

from ._helpers import pattern_suite


# tiny body with long shadows on both sides
BULL = [
    (99.9, 101.5, 98.5, 100.0),
]

globals().update(
    pattern_suite(
        name="cdl_highwave",
        fn=cdl_highwave,
        polars_fn=cdl_highwave_polars,
        output_col="CDL_HIGHWAVE",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
        extra={"strict": False, "symmetric": False},
    )
)
