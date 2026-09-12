# -*- coding: utf-8 -*-
"""Tests for Shooting Star (cdl_shootingstar)."""

from ta.src.candle.cdl_shootingstar import (
    cdl_shootingstar,
    cdl_shootingstar_polars,
)

from ._helpers import pattern_suite


# small body low in the range, long top wick
BULL = [
    (100.0, 103.0, 99.9, 100.3),
]

globals().update(
    pattern_suite(
        name="cdl_shootingstar",
        fn=cdl_shootingstar,
        polars_fn=cdl_shootingstar_polars,
        output_col="CDL_SHOOTINGSTAR",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
