# -*- coding: utf-8 -*-
"""Tests for Long Line Candle (cdl_longline)."""

from ta.src.candle.cdl_longline import cdl_longline, cdl_longline_polars

from ._helpers import pattern_suite


# long real body with short shadows
BULL = [
    (100.0, 102.5, 99.5, 102.3),
]

globals().update(
    pattern_suite(
        name="cdl_longline",
        fn=cdl_longline,
        polars_fn=cdl_longline_polars,
        output_col="CDL_LONGLINE",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
