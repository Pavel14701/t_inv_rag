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
def _cdl_breakaway_nb(
    open_, high, low, close,
    min_body_factor,
    max_shadow_factor,
    strict,
    symmetric
):
    """
    Numba-accelerated Breakaway pattern with optional strict filtering
    and optional symmetric mode.

    Returns:
        1.0 → bullish breakaway
       -1.0 → bearish breakaway
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(4, n):
        # Candle 1
        o4, c4 = open_[i-4], close[i-4]
        h4, l4 = high[i-4], low[i-4]

        # Candle 2
        o3, c3 = open_[i-3], close[i-3]
        h3, l3 = high[i-3], low[i-3]

        # Candle 3
        o2, c2 = open_[i-2], close[i-2]
        h2, l2 = high[i-2], low[i-2]

        # Candle 4
        o1, c1 = open_[i-1], close[i-1]
        h1, l1 = high[i-1], low[i-1]

        # Candle 5 (signal)
        o0, c0 = open_[i], close[i]
        h0, l0 = high[i], low[i]

        direction = 0.0

        # ---------------- Bearish Breakaway (TA-Lib canonical) ----------------
        bear = (
            (c4 > o4) and (c3 > o3) and (c2 > o2) and (c1 > o1) and   # uptrend
            (o3 > h4) and                                             # gap up
            (c0 < o0) and                                             # bearish final candle
            (c0 < c1) and                                             # closes below previous close
            (c0 < c4)                                                 # closes into the gap
        )

        # ---------------- Bullish Breakaway (TA-Lib canonical) ----------------
        bull = (
            (c4 < o4) and (c3 < o3) and (c2 < o2) and (c1 < o1) and   # downtrend
            (o3 < l4) and                                             # gap down
            (c0 > o0) and                                             # bullish final candle
            (c0 > c1) and                                             # closes above previous close
            (c0 > c4)                                                 # closes into the gap
        )

        # ---------------- Mirrored symmetric mode ----------------
        if symmetric:
            # symmetric bullish = canonical bullish (already included)
            # symmetric bearish = canonical bearish (already included)
            pass

        if bull:
            direction = 1.0
        elif bear:
            direction = -1.0
        else:
            continue

        # ---------------- Strict filters ----------------
        if strict:
            # ranges
            r4 = h4 - l4
            r3 = h3 - l3
            r2 = h2 - l2
            r1 = h1 - l1
            r0 = h0 - l0

            # bodies
            b4 = abs(c4 - o4)
            b3 = abs(c3 - o3)
            b2 = abs(c2 - o2)
            b1 = abs(c1 - o1)
            b0 = abs(c0 - o0)

            # minimum body size
            if (b4 < min_body_factor * r4 or
                b3 < min_body_factor * r3 or
                b2 < min_body_factor * r2 or
                b1 < min_body_factor * r1 or
                b0 < min_body_factor * r0):
                continue

            # shadows (fast)
            def shadow(o, c, h, l):
                up = o if o > c else c
                lo = c if o > c else o
                return (h - up) + (lo - l)

            sh4 = shadow(o4, c4, h4, l4)
            sh3 = shadow(o3, c3, h3, l3)
            sh2 = shadow(o2, c2, h2, l2)
            sh1 = shadow(o1, c1, h1, l1)
            sh0 = shadow(o0, c0, h0, l0)

            if (sh4 > max_shadow_factor * r4 or
                sh3 > max_shadow_factor * r3 or
                sh2 > max_shadow_factor * r2 or
                sh1 > max_shadow_factor * r1 or
                sh0 > max_shadow_factor * r0):
                continue

        out[i] = direction

    return out


def cdl_breakaway(
    open_, high, low, close,
    offset=0,
    fillna=None,
    use_talib=True,
    strict=False,
    symmetric=False,
    min_body_factor=0.0,
    max_shadow_factor=1.0,
):
    """
    Universal Breakaway pattern with strict mode and optional symmetric mode.

    If symmetric=False and TA-Lib is available → TA-Lib is used.
    If symmetric=True → TA-Lib is skipped and Numba is always used.
    """
    # Polars → NumPy
    if isinstance(open_, pl.Series): open_ = open_.to_numpy()
    if isinstance(high, pl.Series): high = high.to_numpy()
    if isinstance(low, pl.Series): low = low.to_numpy()
    if isinstance(close, pl.Series): close = close.to_numpy()

    # float64 + contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if not open_.flags.c_contiguous: open_ = np.ascontiguousarray(open_)
    if not high.flags.c_contiguous: high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous: low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous: close = np.ascontiguousarray(close)

    # TA-Lib branch (only if symmetric=False)
    if use_talib and talib_available and not symmetric:
        talib_out = talib.CDLBREAKAWAY(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    # Numba branch
    out = _cdl_breakaway_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_breakaway_polars(
    df: pl.DataFrame,
    open_col="open",
    high_col="high",
    low_col="low",
    close_col="close",
    offset=0,
    fillna=None,
    strict=False,
    symmetric=False,
    min_body_factor=0.0,
    max_shadow_factor=1.0,
    output_col="CDL_BREAKAWAY",
):
    out = cdl_breakaway(
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
