# -*- coding: utf-8 -*-
"""Tests for Three Black Crows (cdl_3blackcrows)."""

from ._helpers import pattern_suite
from ....candle.cdl_3blackcrows import (
    cdl_3blackcrows,
    cdl_3blackcrows_polars,
)

# three black candles, each opening inside the previous body,
# closes falling successively
BULL = [
    (103.0, 103.5, 99.5, 100.0),
    (102.0, 102.5, 98.5, 99.0),
    (101.0, 101.5, 97.5, 98.0),
]

globals().update(pattern_suite(
    name='cdl_3blackcrows',
    fn=cdl_3blackcrows,
    polars_fn=cdl_3blackcrows_polars,
    output_col='CDL_3BLACKCROWS',
    bull=BULL,
    bull_value=1.0,
    talib_values=(0.0, 1.0),
))
