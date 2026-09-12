# -*- coding: utf-8 -*-
"""Tests for Long-Legged Doji (cdl_longleggeddoji)."""

from ta.src.candle.cdl_longleggeddoji import (
    cdl_longleggeddoji,
    cdl_longleggeddoji_polars,
)

from ._helpers import pattern_suite


# near-zero body with long shadows on both sides
BULL = [
    (100.0, 102.0, 98.0, 100.05),
]

globals().update(
    pattern_suite(
        name="cdl_longleggeddoji",
        fn=cdl_longleggeddoji,
        polars_fn=cdl_longleggeddoji_polars,
        output_col="CDL_LONGLEGGEDDOJI",
        bull=BULL,
        bull_value=1.0,
        talib_values=(0.0, 1.0),
    )
)
