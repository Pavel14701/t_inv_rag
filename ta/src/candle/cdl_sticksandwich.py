# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, njit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (float64[:], float64[:], float64[:], float64[:]),
    nopython=True,
    cache=True
)
def _cdl_sticksandwich_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba-accelerated Stick Sandwich pattern.
    Returns boolean mask where pattern completes (True at the 3rd candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)

    for i in range(2, n):
        # Candle 1 (i-2): black
        o1 = open_[i - 2]
        c1 = close[i - 2]
        h1 = high[i - 2]
        l1 = low[i - 2]

        if not (c1 < o1):
            continue

        rng1 = h1 - l1
        body1 = o1 - c1
        if rng1 <= 0.0 or body1 < 0.4 * rng1:
            continue  # reasonably long black

        # Candle 2 (i-1): white
        o2 = open_[i - 1]
        c2 = close[i - 1]
        h2 = high[i - 1]
        l2 = low[i - 1]

        rng2 = h2 - l2
        body2 = c2 - o2
        if rng2 <= 0.0 or body2 <= 0.0:
            continue
        if not (c2 > o2):
            continue  # must be bullish

        # body of candle 2 inside body of candle 1
        if not (c2 < o1 and c2 > c1):
            continue
        if not (o2 < o1 and o2 > c1):
            continue

        # Candle 3 (i): black, close ≈ close1
        o3 = open_[i]
        c3 = close[i]
        h3 = high[i]
        l3 = low[i]

        if not (c3 < o3):
            continue

        rng3 = h3 - l3
        body3 = o3 - c3
        if rng3 <= 0.0 or body3 < 0.4 * rng3:
            continue  # reasonably long black

        # closes of 1st and 3rd nearly equal
        if abs(c3 - c1) > 1e-12:
            continue

        out[i] = True

    return out


def cdl_sticksandwich(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Stick Sandwich pattern.
    Returns numpy array of float64: 1.0 where pattern occurs, else 0.0.
    """
    if isinstance(open_, pl.Series): 
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series): 
        high = high.to_numpy()
    if isinstance(low, pl.Series): 
        low = low.to_numpy()
    if isinstance(close, pl.Series): 
        close = close.to_numpy()
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    if use_talib and talib_available:
        talib_out = talib.CDLSTICKSANDWICH(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    mask = _cdl_sticksandwich_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_sticksandwich_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_STICKSANDWICH",
) -> pl.DataFrame:
    """
    Add Stick Sandwich column to Polars DataFrame.
    """
    out = cdl_sticksandwich(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
