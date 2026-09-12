# -*- coding: utf-8 -*-
"""Tests for Harami Cross (cdl_haramicross)."""

from ta.src.candle.cdl_haramicross import (
    cdl_haramicross,
    cdl_haramicross_polars,
)

from ._helpers import pattern_suite


# large black body, doji inside it
BULL = [
    (103.0, 103.5, 99.5, 100.0),
    (101.0, 103.0, 100.5, 101.05),
]

# large white body, doji inside it
BEAR = [
    (100.0, 100.5, 99.5, 103.0),
    (102.0, 103.5, 101.5, 101.85),
]

globals().update(
    pattern_suite(
        name="cdl_haramicross",
        fn=cdl_haramicross,
        polars_fn=cdl_haramicross_polars,
        output_col="CDL_HARAMICROSS",
        bull=BULL,
        bear=BEAR,
        bull_value=1.0,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
        extra={"strict": False},
    )
)
