# -*- coding: utf-8 -*-
"""Tests for On-Neck (cdl_onneck)."""

from ta.src.candle.cdl_onneck import cdl_onneck, cdl_onneck_polars

from ._helpers import pattern_suite


# long black, gap-down white closing back at the prior close level
BULL = [
    (103.0, 103.5, 99.5, 100.0),
    (99.2, 100.5, 99.0, 100.2),
]

globals().update(
    pattern_suite(
        name="cdl_onneck",
        fn=cdl_onneck,
        polars_fn=cdl_onneck_polars,
        output_col="CDL_ONNECK",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
