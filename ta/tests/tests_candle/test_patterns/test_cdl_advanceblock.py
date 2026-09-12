# -*- coding: utf-8 -*-
"""Tests for Advance Block (cdl_advanceblock)."""

from ta.src.candle.cdl_advanceblock import (
    cdl_advanceblock,
    cdl_advanceblock_polars,
)

from ._helpers import pattern_suite


# three rising whites with shrinking bodies -> bearish Advance Block
BEAR = [
    (100.0, 100.5, 99.5, 103.5),
    (103.8, 104.3, 103.3, 106.3),
    (106.6, 107.1, 106.1, 108.1),
]

globals().update(
    pattern_suite(
        name="cdl_advanceblock",
        fn=cdl_advanceblock,
        polars_fn=cdl_advanceblock_polars,
        output_col="CDL_ADVANCEBLOCK",
        bear=BEAR,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
        extra={"strict": False, "symmetric": False},
    )
)
