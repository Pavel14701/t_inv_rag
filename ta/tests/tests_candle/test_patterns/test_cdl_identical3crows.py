# -*- coding: utf-8 -*-
"""Tests for Identical Three Crows (cdl_identical3crows)."""

from ta.src.candle.cdl_identical3crows import (
    cdl_identical3crows,
    cdl_identical3crows_polars,
)

from ._helpers import pattern_suite


# three black candles with falling closes, small lower shadows,
# each opening near the previous close
BULL = [
    (105.0, 105.2, 103.0, 103.0),
    (103.1, 103.3, 100.8, 101.0),
    (101.1, 101.3, 98.8, 99.0),
]

globals().update(
    pattern_suite(
        name="cdl_identical3crows",
        fn=cdl_identical3crows,
        polars_fn=cdl_identical3crows_polars,
        output_col="CDL_IDENTICAL3CROWS",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
