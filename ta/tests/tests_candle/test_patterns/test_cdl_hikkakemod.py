# -*- coding: utf-8 -*-
"""Tests for Hikkake Modified (cdl_hikkakemod)."""

from ta.src.candle.cdl_hikkakemod import cdl_hikkakemod, cdl_hikkakemod_polars

from ._helpers import pattern_suite


# inside bar, downward break, then close back above the inside-bar high
# (bullish hikkake modified). Detection at index 6 (prefix len 4 + 2).
BULL = [
    (100.0, 103.0, 97.0, 101.0),
    (101.0, 102.0, 98.0, 100.5),
    (100.5, 101.8, 97.2, 97.5),
    (97.6, 102.4, 97.2, 102.2),
    (102.1, 103.0, 101.5, 102.5),
    (102.4, 103.2, 101.8, 102.8),
]

globals().update(
    pattern_suite(
        name="cdl_hikkakemod",
        fn=cdl_hikkakemod,
        polars_fn=cdl_hikkakemod_polars,
        output_col="CDL_HIKKAKEMOD",
        bull=BULL,
        bull_value=1.0,
        bull_idx=6,
        talib_values=(0.0, 1.0, -1.0, 2.0, -2.0),
        extra={"strict": False, "lookahead": 3},
    )
)
