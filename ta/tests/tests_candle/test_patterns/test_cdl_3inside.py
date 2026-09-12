# -*- coding: utf-8 -*-
"""Tests for Three Inside Up/Down (cdl_3inside)."""

from ta.src.candle.cdl_3inside import cdl_3inside, cdl_3inside_polars

from ._helpers import pattern_suite


# bearish long, bullish harami inside it, bullish close above harami
BULL = [
    (102.0, 102.5, 99.5, 100.0),
    (100.5, 101.8, 100.0, 101.5),
    (101.6, 102.8, 101.2, 102.5),
]

# bullish long, bearish harami inside it, bearish close below harami
BEAR = [
    (100.0, 100.5, 99.5, 102.0),
    (102.5, 103.0, 99.3, 99.5),
    (99.0, 99.4, 98.0, 98.0),
]

globals().update(
    pattern_suite(
        name="cdl_3inside",
        fn=cdl_3inside,
        polars_fn=cdl_3inside_polars,
        output_col="CDL_3INSIDE",
        bull=BULL,
        bear=BEAR,
        bull_value=1.0,
        bear_value=-1.0,
        talib_values=(0.0, 1.0, -1.0),
    )
)
