# -*- coding: utf-8 -*-
"""Tests for Short Line Candle (cdl_shortline).

The kernel follows TA-Lib semantics: body and shadows must each be smaller
than 0.3x the corresponding 5-bar averages.
"""

from ._helpers import pattern_suite
from ....candle.cdl_shortline import cdl_shortline, cdl_shortline_polars

# a long candle followed by a much shorter one (averages are dominated by
# the long candle and the flat prefix), detection at index 6 (5 flat + 2)
BULL = [
    (100.0, 103.5, 99.5, 103.0),
    (100.0, 100.2, 99.99, 100.1),
]

globals().update(pattern_suite(
    name='cdl_shortline',
    fn=cdl_shortline,
    polars_fn=cdl_shortline_polars,
    output_col='CDL_SHORTLINE',
    bull=BULL,
    bull_value=1.0,
    bull_idx=6,
    talib_values=(0.0, 1.0),
    prefix_n=5,
))
