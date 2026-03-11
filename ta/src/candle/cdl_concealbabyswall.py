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
def _cdl_concealbabyswall_nb(
    open_, high, low, close,
    min_body_factor,
    max_shadow_factor,
    strict,
    symmetric
):
    """
    Optimized Concealing Baby Swallow pattern.

    Returns:
        1.0 → bullish concealing baby swallow
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(3, n):
        o3 = open_[i-3]
        c3 = close[i-3]
        h3 = high[i-3]
        l3 = low[i-3]

        o2 = open_[i-2]
        c2 = close[i-2]
        h2 = high[i-2]
        l2 = low[i-2]

        o1 = open_[i-1]
        c1 = close[i-1]
        h1 = high[i-1]
        l1 = low[i-1]

        o0 = open_[i]
        c0 = close[i]
        h0 = high[i]
        l0 = low[i]

        r3 = h3 - l3
        r2 = h2 - l2
        r1 = h1 - l1
        r0 = h0 - l0
        if r3 <= 0.0 or r2 <= 0.0 or r1 <= 0.0 or r0 <= 0.0:
            continue

        # First two candles: long black, no upper shadow
        if not (c3 < o3 and c2 < o2):
            continue
        if not (c3 == h3 and c2 == h2):
            continue

        # Third and fourth: black, gap and swallow logic
        if not (c1 < o1 and c0 < o0):
            continue

        # Candle 3 opens within body of candle 2 and closes below
        if not (o1 < o2 and o1 > c2 and c1 < c2):
            continue

        # Candle 4 opens within body of candle 3 and closes below
        if not (o0 < o1 and o0 > c1 and c0 < c1):
            continue

        if strict:
            b3 = o3 - c3
            b2 = o2 - c2
            if b3 < min_body_factor * r3 or b2 < min_body_factor * r2:
                continue

            up3 = o3 if o3 > c3 else c3
            lo3 = c3 if o3 > c3 else o3
            sh3 = (h3 - up3) + (lo3 - l3)

            up2 = o2 if o2 > c2 else c2
            lo2 = c2 if o2 > c2 else o2
            sh2 = (h2 - up2) + (lo2 - l2)

            up1 = o1 if o1 > c1 else c1
            lo1 = c1 if o1 > c1 else o1
            sh1 = (h1 - up1) + (lo1 - l1)

            up0 = o0 if o0 > c0 else c0
            lo0 = c0 if o0 > c0 else o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if (sh3 > max_shadow_factor * r3 or
                sh2 > max_shadow_factor * r2 or
                sh1 > max_shadow_factor * r1 or
                sh0 > max_shadow_factor * r0):
                continue

        out[i] = 1.0

    return out


def cdl_concealbabyswall(
    open_, high, low, close,
    offset=0,
    fillna=None,
    use_talib=True,
    strict=False,
    symmetric=False,
    min_body_factor=0.5,
    max_shadow_factor=0.3,
):
    """
    Concealing Baby Swallow pattern with strict support.

    TA-Lib has only bullish variant; symmetric flag is kept for API consistency
    but does not enable a bearish mirror.
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
        talib_out = talib.CDLCONCEALBABYSWALL(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    # Numba branch
    out = _cdl_concealbabyswall_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_concealbabyswall_polars(
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
    max_shadow_factor=0.3,
    output_col="CDL_CONCEALBABYSWALL",
):
    out = cdl_concealbabyswall(
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
