# -*- coding: utf-8 -*-
"""Tests for Two Crows (cdl_2crows)."""

from ta.src.candle.cdl_2crows import cdl_2crows, cdl_2crows_polars

from ._helpers import pattern_suite


# white long, then black opening above its close, then another black
# closing below the second's close -> bearish reversal
BULL = [
    (100.0, 102.5, 99.5, 102.0),
    (103.0, 103.5, 100.5, 101.0),
    (102.5, 103.2, 100.0, 100.5),
]

globals().update(
    pattern_suite(
        name="cdl_2crows",
        fn=cdl_2crows,
        polars_fn=cdl_2crows_polars,
        output_col="CDL_2CROWS",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
