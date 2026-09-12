# -*- coding: utf-8 -*-
"""Tests for Inverted Hammer (cdl_invertedhammer)."""

from ta.src.candle.cdl_invertedhammer import (
    cdl_invertedhammer,
    cdl_invertedhammer_polars,
)

from ._helpers import pattern_suite


# small body near the low with a long upper shadow
BULL = [
    (100.0, 103.0, 100.4, 100.5),
]

globals().update(
    pattern_suite(
        name="cdl_invertedhammer",
        fn=cdl_invertedhammer,
        polars_fn=cdl_invertedhammer_polars,
        output_col="CDL_INVERTEDHAMMER",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
