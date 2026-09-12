# -*- coding: utf-8 -*-
"""Tests for Piercing pattern (cdl_piercing)."""

from ta.src.candle.cdl_piercing import cdl_piercing, cdl_piercing_polars

from ._helpers import pattern_suite


# long black, gap-down open, white close above the first midpoint
# first body but below its open
BULL = [
    (102.0, 102.5, 99.5, 100.0),
    (99.0, 102.2, 98.5, 101.5),
]

globals().update(
    pattern_suite(
        name="cdl_piercing",
        fn=cdl_piercing,
        polars_fn=cdl_piercing_polars,
        output_col="CDL_PIERCING",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
