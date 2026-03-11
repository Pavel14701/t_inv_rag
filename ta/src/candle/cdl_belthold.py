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
def _cdl_belthold_nb(
    open_, high, low, close,
    min_body_factor,
    max_shadow_factor,
    strict,
    symmetric
):
    """
    Numba-accelerated Belt Hold pattern with optional strict filtering
    and optional symmetric mode.

    Returns:
        1.0 → bullish belt hold
       -1.0 → bearish belt hold
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        o0, c0 = open_[i], close[i]
        h0, l0 = high[i], low[i]

        direction = 0.0

        # ---------------- Bullish Belt Hold ----------------
        # Long bullish candle with no lower shadow (open == low)
        bull = (
            (c0 > o0) and
            (o0 == l0) and
            ((c0 - o0) > (h0 - l0) * 0.5)  # long body
        )

        # ---------------- Bearish Belt Hold ----------------
        # Long bearish candle with no upper shadow (open == high)
        bear = (
            (c0 < o0) and
            (o0 == h0) and
            ((o0 - c0) > (h0 - l0) * 0.5)
        )

        if bull:
            direction = 1.0
        elif bear:
            direction = -1.0
        else:
            continue

        # ---------------- Strict filters ----------------
        if strict:
            r0 = h0 - l0
            b0 = abs(c0 - o0)

            # Minimum body size
            if b0 < min_body_factor * r0:
                continue

            # Shadow calculation (fast)
            up0 = o0 if o0 > c0 else c0
            lo0 = c0 if o0 > c0 else o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if sh0 > max_shadow_factor * r0:
                continue

        out[i] = direction

    return out


def cdl_belthold(
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
    Universal Belt Hold pattern with strict mode and optional symmetric mode.

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
        talib_out = talib.CDLBELTHOLD(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    # Numba branch
    out = _cdl_belthold_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_belthold_polars(
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
    output_col="CDL_BELTHOLD",
):
    out = cdl_belthold(
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
