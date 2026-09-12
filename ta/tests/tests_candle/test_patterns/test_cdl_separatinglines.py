# -*- coding: utf-8 -*-
"""Tests for Separating Lines (cdl_separatinglines)."""

from ta.src.candle.cdl_separatinglines import (
    cdl_separatinglines,
    cdl_separatinglines_polars,
)

from ._helpers import pattern_suite


# black candle, then white with the same open, closing higher
BULL = [
    (102.0, 102.5, 99.0, 100.0),
    (102.0, 103.0, 101.9, 102.5),
]

globals().update(
    pattern_suite(
        name="cdl_separatinglines",
        fn=cdl_separatinglines,
        polars_fn=cdl_separatinglines_polars,
        output_col="CDL_SEPARATINGLINES",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
