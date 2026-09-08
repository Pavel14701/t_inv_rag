# -*- coding: utf-8 -*-
"""Tests for Evening Star (cdl_eveningstar)."""

from ._helpers import pattern_suite
from ....candle.cdl_eveningstar import cdl_eveningstar, cdl_eveningstar_polars

# long white, small-bodied star gapped up, black closing below the midpoint
BEAR = [
    (100.0, 100.5, 99.5, 103.0),
    (103.5, 104.2, 103.2, 103.6),
    (103.8, 104.0, 100.5, 101.0),
]

globals().update(pattern_suite(
    name='cdl_eveningstar',
    fn=cdl_eveningstar,
    polars_fn=cdl_eveningstar_polars,
    output_col='CDL_EVENINGSTAR',
    bear=BEAR,
    bear_value=-1.0,
    talib_values=(0.0, 1.0, -1.0),
    extra={'strict': False},
))
