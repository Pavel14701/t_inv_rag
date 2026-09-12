# -*- coding: utf-8 -*-
"""Tests for Hikkake (cdl_hikkake)."""

from ta.src.candle.cdl_hikkake import cdl_hikkake, cdl_hikkake_polars

from ._helpers import pattern_suite


# inside bar, upward break, then close back below the inside-bar low
# (bearish hikkake / bull trap). Detection at index 6 (prefix len 4 + 2).
BULL = [
    (100.0, 103.0, 97.0, 101.0),
    (101.0, 102.0, 98.0, 100.5),
    (101.5, 102.5, 98.2, 101.0),
    (100.8, 101.0, 97.5, 97.0),
    (97.2, 98.0, 96.5, 96.8),
    (96.9, 97.5, 96.0, 96.2),
]

globals().update(
    pattern_suite(
        name="cdl_hikkake",
        fn=cdl_hikkake,
        polars_fn=cdl_hikkake_polars,
        output_col="CDL_HIKKAKE",
        bull=BULL,
        bull_value=-1.0,
        bull_idx=6,
        # TA-Lib CDLHIKKAKE also returns +-200 (confirmation) -> +-2 after /100
        talib_values=(0.0, 1.0, -1.0, 2.0, -2.0),
        extra={"strict": False, "lookahead": 3},
    )
)
