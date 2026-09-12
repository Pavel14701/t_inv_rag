# -*- coding: utf-8 -*-
"""Tests for Homing Pigeon (cdl_homingpigeon)."""

from ta.src.candle.cdl_homingpigeon import (
    cdl_homingpigeon,
    cdl_homingpigeon_polars,
)

from ._helpers import pattern_suite


# white candle, then a smaller white body contained in it
BULL = [
    (102.0, 102.5, 99.5, 100.0),
    (101.0, 101.5, 100.2, 100.5),
]

globals().update(
    pattern_suite(
        name="cdl_homingpigeon",
        fn=cdl_homingpigeon,
        polars_fn=cdl_homingpigeon_polars,
        output_col="CDL_HOMINGPIGEON",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
