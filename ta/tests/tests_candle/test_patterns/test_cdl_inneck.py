# -*- coding: utf-8 -*-
"""Tests for Inneck (cdl_inneck)."""

from ta.src.candle.cdl_inneck import cdl_inneck, cdl_inneck_polars

from ._helpers import pattern_suite


# black candle, gap-down white closing back up toward the previous close
BULL = [
    (103.0, 103.5, 100.0, 100.5),
    (99.8, 101.3, 99.5, 100.9),
]

globals().update(
    pattern_suite(
        name="cdl_inneck",
        fn=cdl_inneck,
        polars_fn=cdl_inneck_polars,
        output_col="CDL_INNECK",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
