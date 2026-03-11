# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import njit, types

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (types.float64[:], types.float64[:], types.float64[:], types.float64[:],
     types.float64, types.float64, types.boolean, types.boolean),
    nopython=True,
    cache=True,
    fastmath=True
)
def _cdl_counterattack_nb(
    open_, high, low, close,
    min_body_factor,
    max_shadow_factor,
    strict,
    symmetric
):
    """
    Optimized Counterattack pattern.

    Returns:
        1.0 → bullish counterattack
       -1.0 → bearish counterattack
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        o1 = open_[i-1]
        c1 = close[i-1]
        h1 = high[i-1]
        l1 = low[i-1]

        o0 = open_[i]
        c0 = close[i]
        h0 = high[i]
        l0 = low[i]

        r1 = h1 - l1
        r0 = h0 - l0
        if r1 <= 0.0 or r0 <= 0.0:
            continue

        # First candle body
        b1 = c1 - o1 if c1 > o1 else o1 - c1
        # Second candle body
        b0 = c0 - o0 if c0 > o0 else o0 - c0

        if b1 <= 0.0 or b0 <= 0.0:
            continue

        # Colors
        bull1 = c1 > o1
        bear1 = c1 < o1
        bull0 = c0 > o0
        bear0 = c0 < o0

        direction = 0.0

        # Bearish counterattack: first bullish, second bearish, gap up, close near previous close
        if bull1 and bear0:
            if o0 > h1 and c0 < c1:
                if abs(c0 - c1) <= 0.25 * (r0 + r1):
                    direction = -1.0

        # Bullish counterattack: first bearish, second bullish, gap down, close near previous close
        elif bear1 and bull0:
            if o0 < l1 and c0 > c1:
                if abs(c0 - c1) <= 0.25 * (r0 + r1):
                    direction = 1.0

        if direction == 0.0:
            continue

        if strict:
            # Minimum body size
            if (b1 < min_body_factor * r1) or (b0 < min_body_factor * r0):
                continue

            # Shadows (fast)
            up1 = o1 if o1 > c1 else c1
            lo1 = c1 if o1 > c1 else o1
            sh1 = (h1 - up1) + (lo1 - l1)

            up0 = o0 if o0 > c0 else c0
            lo0 = c0 if o0 > c0 else o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if sh1 > max_shadow_factor * r1 or sh0 > max_shadow_factor * r0:
                continue

        out[i] = direction

    return out


def cdl_counterattack(
    open_, high, low, close,
    offset=0,
    fillna=None,
    use_talib=True,
    strict=False,
    symmetric=False,
    min_body_factor=0.3,
    max_shadow_factor=0.5,
):
    """
    Counterattack pattern with strict and symmetric support.
    """
    if isinstance(open_, pl.Series): open_ = open_.to_numpy()
    if isinstance(high, pl.Series): high = high.to_numpy()
    if isinstance(low, pl.Series): low = low.to_numpy()
    if isinstance(close, pl.Series): close = close.to_numpy()

    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if use_talib and talib_available and not symmetric:
        talib_out = talib.CDLCOUNTERATTACK(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    out = _cdl_counterattack_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_counterattack_polars(
    df: pl.DataFrame,
    open_col="open",
    high_col="high",
    low_col="low",
    close_col="close",
    offset=0,
    fillna=None,
    strict=False,
    symmetric=False,
    min_body_factor=0.3,
    max_shadow_factor=0.5,
    output_col="CDL_COUNTERATTACK",
):
    out = cdl_counterattack(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
        strict=strict,
        symmetric=symmetric,
        min_body_factor=min_body_factor,
        max_shadow_factor=max_shadow_factor,
    )
    return df.with_columns(pl.Series(output_col, out))

