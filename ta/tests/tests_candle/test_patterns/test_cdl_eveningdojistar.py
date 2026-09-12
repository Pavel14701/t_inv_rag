# -*- coding: utf-8 -*-
"""Tests for Evening Doji Star (cdl_eveningdojistar)."""

from ta.src.candle.cdl_eveningdojistar import (
    cdl_eveningdojistar,
    cdl_eveningdojistar_polars,
)

from ._helpers import pattern_suite


# long white, gapped-up doji, black closing below the midpoint of the first
BEAR = [
    (100.0, 100.5, 99.5, 103.0),
    (103.5, 104.2, 103.2, 103.45),
    (103.8, 104.0, 100.5, 101.0),
]

globals().update(
    pattern_suite(
        name="cdl_eveningdojistar",
        fn=cdl_eveningdojistar,
        polars_fn=cdl_eveningdojistar_polars,
        output_col="CDL_EVENINGDOJISTAR",
        bear=BEAR,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
        extra={"strict": False},
    )
)
