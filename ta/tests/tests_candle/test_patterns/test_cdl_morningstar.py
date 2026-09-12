# -*- coding: utf-8 -*-
"""Tests for Morning Star (cdl_morningstar)."""

from ta.src.candle.cdl_morningstar import (
    cdl_morningstar,
    cdl_morningstar_polars,
)

from ._helpers import pattern_suite


# long black, small-bodied star gapped down, long white above the midpoint
BULL = [
    (103.0, 103.4, 99.4, 100.0),
    (98.2, 99.5, 98.0, 99.3),
    (99.5, 102.5, 99.2, 102.3),
]

globals().update(
    pattern_suite(
        name="cdl_morningstar",
        fn=cdl_morningstar,
        polars_fn=cdl_morningstar_polars,
        output_col="CDL_MORNINGSTAR",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
