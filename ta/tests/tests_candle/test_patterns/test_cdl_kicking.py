# -*- coding: utf-8 -*-
"""Tests for Kicking (cdl_kicking)."""

from ta.src.candle.cdl_kicking import cdl_kicking, cdl_kicking_polars

from ._helpers import pattern_suite


# black marubozu, then white marubozu gapping above its body
BULL = [
    (103.0, 103.2, 100.1, 100.1),
    (103.5, 106.5, 103.4, 106.3),
]

globals().update(
    pattern_suite(
        name="cdl_kicking",
        fn=cdl_kicking,
        polars_fn=cdl_kicking_polars,
        output_col="CDL_KICKING",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
