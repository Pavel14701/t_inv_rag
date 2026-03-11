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
def _cdl_closingmarubozu_nb(
    open_, high, low, close,
    min_body_factor,
    max_shadow_factor,
    strict,
    symmetric
):
    """
    Optimized Closing Marubozu pattern.
    Returns:
        1.0 → bullish closing marubozu
       -1.0 → bearish closing marubozu
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        o = open_[i]
        h = high[i]
        l = low[i]
        c = close[i]

        rng = h - l
        if rng <= 0.0:
            continue

        # Bullish: close == high
        bull = (c > o) and (c == h)

        # Bearish: close == low
        bear = (c < o) and (c == l)

        if bull:
            direction = 1.0
        elif bear:
            direction = -1.0
        else:
            continue

        if strict:
            body = c - o if c > o else o - c

            # Body must be large enough
            if body < min_body_factor * rng:
                continue

            # Shadows (fast, no max/min)
            up = o if o > c else c
            lo = c if o > c else o
            shadow = (h - up) + (lo - l)

            if shadow > max_shadow_factor * rng:
                continue

        out[i] = direction

    return out


def cdl_closingmarubozu(
    open_, high, low, close,
    offset=0,
    fillna=None,
    use_talib=True,
    strict=False,
    symmetric=False,
    min_body_factor=0.5,
    max_shadow_factor=0.2,
):
    """
    Closing Marubozu with strict and symmetric support.
    """
    # Polars → NumPy
    if isinstance(open_, pl.Series): open_ = open_.to_numpy()
    if isinstance(high, pl.Series): high = high.to_numpy()
    if isinstance(low, pl.Series): low = low.to_numpy()
    if isinstance(close, pl.Series): close = close.to_numpy()

    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # TA-Lib branch (only if symmetric=False)
    if use_talib and talib_available and not symmetric:
        talib_out = talib.CDLCLOSINGMARUBOZU(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    # Numba branch
    out = _cdl_closingmarubozu_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_closingmarubozu_polars(
    df: pl.DataFrame,
    open_col="open",
    high_col="high",
    low_col="low",
    close_col="close",
    offset=0,
    fillna=None,
    strict=False,
    symmetric=False,
    min_body_factor=0.5,
    max_shadow_factor=0.2,
    output_col="CDL_CLOSINGMARUBOZU",
):
    out = cdl_closingmarubozu(
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
