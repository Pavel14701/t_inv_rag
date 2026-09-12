# -*- coding: utf-8 -*-
"""Tests for Marubozu (cdl_marubozu)."""

from ta.src.candle.cdl_marubozu import cdl_marubozu, cdl_marubozu_polars

from ._helpers import pattern_suite


# (almost) no shadows, body ≈ range
BULL = [
    (100.0, 103.0, 99.9, 102.95),
]

globals().update(
    pattern_suite(
        name="cdl_marubozu",
        fn=cdl_marubozu,
        polars_fn=cdl_marubozu_polars,
        output_col="CDL_MARUBOZU",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
