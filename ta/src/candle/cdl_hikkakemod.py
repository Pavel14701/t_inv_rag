# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from numba import njit, types

from .._array_ops import _apply_offset_fillna
from ..external import talib, talib_available


@njit(
    (
        types.float64[:],
        types.float64[:],
        types.float64[:],
        types.float64[:],
        types.int64,
        types.boolean,
    ),
    cache=True,
    fastmath=False,
)
def _cdl_hikkakemod_nb(
    open_,
    high,
    low,
    close,
    lookahead,
    strict,
):
    """Optimized Hikkake Modified pattern.

    Returns:
        1.0 -> bullish hikkake modified
       -1.0 -> bearish hikkake modified
        0.0 -> none

    """
    n = len(open_)
    out = np.zeros(n, np.float64)
    for i in range(2, n - lookahead):
        # Bar -2, -1, 0
        h2 = high[i - 2]
        l2 = low[i - 2]
        h1 = high[i - 1]
        l1 = low[i - 1]
        h0 = high[i]
        l0 = low[i]
        close[i]
        # Inside bar: bar -1 inside bar -2
        if not (h1 < h2 and l1 > l2):
            continue
        # Bar 0 breakout
        bullish_break = h0 > h1 and l0 >= l1
        bearish_break = l0 < l1 and h0 <= h1
        if not (bullish_break or bearish_break):
            continue
        direction = 0.0
        # Modified confirmation: must CLOSE beyond inside bar boundary
        if bullish_break:
            # breakout up -> look for close below inside bar low
            for k in range(1, lookahead + 1):
                if close[i + k] < l1:
                    direction = -1.0
                    break
        elif bearish_break:
            # breakout down -> look for close above inside bar high
            for k in range(1, lookahead + 1):
                if close[i + k] > h1:
                    direction = 1.0
                    break
        if direction == 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
            continue
        if strict:
            # Inside bar must be meaningful
            rng2 = h2 - l2
            rng1 = h1 - l1
            if rng2 <= 0.0 or rng1 <= 0.0:
                continue
            if rng1 < 0.25 * rng2:
                continue
        out[i] = direction
    return out


def cdl_hikkakemod(
    open_,
    high,
    low,
    close,
    offset=0,
    fillna=None,
    use_talib=True,
    strict=False,
    lookahead=3,
):
    """Hikkake Modified pattern with strict support."""
    # Polars -> numpy
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    open_ = np.asarray(open_, np.float64)
    high = np.asarray(high, np.float64)
    low = np.asarray(low, np.float64)
    close = np.asarray(close, np.float64)
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not open_.flags.writeable:
        open_ = open_.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not low.flags.writeable:
        low = low.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    # TA-Lib fallback
    if use_talib and talib_available:
        out = (
            talib.CDLHIKKAKEMOD(open_, high, low, close).astype(np.float64)
            / 100.0
        )
        return _apply_offset_fillna(out, offset, fillna)
    out = _cdl_hikkakemod_nb(
        open_,
        high,
        low,
        close,
        int(lookahead),
        strict,
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_hikkakemod_polars(
    df: pl.DataFrame,
    open_col="open",
    high_col="high",
    low_col="low",
    close_col="close",
    offset=0,
    fillna=None,
    strict=False,
    lookahead=3,
    output_col="CDL_HIKKAKEMOD",
):
    """See module docs."""
    out = cdl_hikkakemod(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
        strict=strict,
        lookahead=lookahead,
    )
    return df.with_columns(pl.Series(output_col, out))
