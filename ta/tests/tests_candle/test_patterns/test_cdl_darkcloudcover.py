# -*- coding: utf-8 -*-
"""Tests for Dark Cloud Cover (cdl_darkcloudcover)."""

from ta.src.candle.cdl_darkcloudcover import (
    cdl_darkcloudcover,
    cdl_darkcloudcover_polars,
)

from ._helpers import pattern_suite


# long white, then black opening above the high and closing inside the body
BEAR = [
    (100.0, 100.5, 99.5, 103.0),
    (103.5, 104.0, 100.8, 101.0),
]

globals().update(
    pattern_suite(
        name="cdl_darkcloudcover",
        fn=cdl_darkcloudcover,
        polars_fn=cdl_darkcloudcover_polars,
        output_col="CDL_DARKCLOUDCOVER",
        bear=BEAR,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
        extra={"strict": False},
    )
)
