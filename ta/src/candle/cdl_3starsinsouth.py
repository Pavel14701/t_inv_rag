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
def _cdl_3starsinsouth_nb(
    open_, high, low, close,
    min_body_factor,
    max_shadow_factor,
    strict,
    symmetric
):
    """
    Three Stars In The South (bullish) + optional mirrored bearish variant.
    1.0 → bullish
   -1.0 → bearish (only if symmetric=True)
    0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(2, n):
        o2, c2 = open_[i - 2], close[i - 2]
        o1, c1 = open_[i - 1], close[i - 1]
        o0, c0 = open_[i], close[i]

        h2, l2 = high[i - 2], low[i - 2]
        h1, l1 = high[i - 1], low[i - 1]
        h0, l0 = high[i], low[i]

        direction = 0.0

        # ---------------- Bullish (TA-Lib canonical) ----------------
        bull = (
            (c2 < o2) and (c1 < o1) and (c0 < o0) and   # три чёрные
            (l1 < l2) and (l0 < l1) and                 # минимумы вниз
            (c0 > c1)                                   # третья закрывается выше второй
        )

        # ---------------- Mirrored bearish (only if symmetric=True) ----------------
        bear = False
        if symmetric:
            bear = (
                (c2 > o2) and (c1 > o1) and (c0 > o0) and   # три белые
                (h1 > h2) and (h0 > h1) and                 # максимумы вверх
                (c0 < c1)   # третья закрывается ниже второй
            )

        if bull:
            direction = 1.0
        elif bear:
            direction = -1.0
        else:
            continue

        # ---------------- Strict filters ----------------
        if strict:
            r2 = h2 - l2
            r1 = h1 - l1
            r0 = h0 - l0

            b2 = abs(c2 - o2)
            b1 = abs(c1 - o1)
            b0 = abs(c0 - o0)

            if (b2 < min_body_factor * r2 or
                b1 < min_body_factor * r1 or
                b0 < min_body_factor * r0):
                continue

            up2 = o2 if o2 > c2 else c2
            lo2 = c2 if o2 > c2 else o2
            sh2 = (h2 - up2) + (lo2 - l2)

            up1 = o1 if o1 > c1 else c1
            lo1 = c1 if o1 > c1 else o1
            sh1 = (h1 - up1) + (lo1 - l1)

            up0 = o0 if o0 > c0 else c0
            lo0 = c0 if o0 > c0 else o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if (sh2 > max_shadow_factor * r2 or
                sh1 > max_shadow_factor * r1 or
                sh0 > max_shadow_factor * r0):
                continue

        out[i] = direction

    return out


def cdl_3starsinsouth(
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
    Three Stars In The South with optional symmetric bearish mode.
    """
    # Polars → NumPy
    if isinstance(open_, pl.Series): 
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series): 
        high = high.to_numpy()
    if isinstance(low, pl.Series): 
        low = low.to_numpy()
    if isinstance(close, pl.Series): 
        close = close.to_numpy()

    # float64 + contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if not open_.flags.c_contiguous: 
        open_ = np.ascontiguousarray(open_)
    if not high.flags.c_contiguous: 
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous: 
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous: 
        close = np.ascontiguousarray(close)

    # TA-Lib branch (only if symmetric=False)
    if use_talib and talib_available and not symmetric:
        talib_out = talib.CDL3STARSINSOUTH(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    # Numba branch
    out = _cdl_3starsinsouth_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_3starsinsouth_polars(
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
    output_col="CDL_3STARSINSOUTH",
):
    out = cdl_3starsinsouth(
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
