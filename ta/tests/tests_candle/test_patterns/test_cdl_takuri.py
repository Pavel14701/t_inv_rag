# -*- coding: utf-8 -*-
"""Tests for Takuri Line (cdl_takuri)."""

from ta.src.candle.cdl_takuri import cdl_takuri, cdl_takuri_polars

from ._helpers import pattern_suite


# tiny body with (almost) no upper shadow and a long lower shadow
BULL = [
    (100.0, 100.05, 98.0, 100.02),
]

globals().update(
    pattern_suite(
        name="cdl_takuri",
        fn=cdl_takuri,
        polars_fn=cdl_takuri_polars,
        output_col="CDL_TAKURI",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
