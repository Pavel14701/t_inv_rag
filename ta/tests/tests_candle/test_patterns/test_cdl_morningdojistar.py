# -*- coding: utf-8 -*-
"""Tests for Morning Doji Star (cdl_morningdojistar)."""

from ta.src.candle.cdl_morningdojistar import (
    cdl_morningdojistar,
    cdl_morningdojistar_polars,
)

from ._helpers import pattern_suite


# long black, gapped-down doji, white closing above the first midpoint
BULL = [
    (103.0, 103.4, 99.4, 100.0),
    (98.25, 99.5, 97.8, 98.2),
    (98.5, 102.3, 98.0, 102.0),
]

globals().update(
    pattern_suite(
        name="cdl_morningdojistar",
        fn=cdl_morningdojistar,
        polars_fn=cdl_morningdojistar_polars,
        output_col="CDL_MORNINGDOJISTAR",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
