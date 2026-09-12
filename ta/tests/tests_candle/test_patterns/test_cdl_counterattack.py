# -*- coding: utf-8 -*-
"""Tests for Counterattack (cdl_counterattack)."""

from ta.src.candle.cdl_counterattack import (
    cdl_counterattack,
    cdl_counterattack_polars,
)

from ._helpers import pattern_suite


# long black, gap-down white closing back near the prior close
BULL = [
    (103.0, 103.5, 100.0, 100.5),
    (99.5, 101.3, 99.0, 100.9),
]

# long white, gap-up black closing back near the prior close
BEAR = [
    (100.0, 103.0, 99.5, 102.5),
    (103.5, 104.0, 101.8, 102.1),
]

globals().update(
    pattern_suite(
        name="cdl_counterattack",
        fn=cdl_counterattack,
        polars_fn=cdl_counterattack_polars,
        output_col="CDL_COUNTERATTACK",
        bull=BULL,
        bear=BEAR,
        bull_value=1.0,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
        extra={"strict": False},
    )
)
