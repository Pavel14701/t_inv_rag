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
def _cdl_tasukigap_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Tasuki Gap pattern.
    Returns boolean mask where pattern completes (True at the 3rd candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # Candle 1
        o1 = open_[i - 2]
        c1 = close[i - 2]
        h1 = high[i - 2]
        l1 = low[i - 2]
        # Candle 2
        o2 = open_[i - 1]
        c2 = close[i - 1]
        h2 = high[i - 1]
        l2 = low[i - 1]
        # Candle 3
        o3 = open_[i]
        c3 = close[i]
        # --- Bullish Tasuki Gap ---
        # Candle1 white, Candle2 white with gap up
        bull = (
            c1 > o1 and
            c2 > o2 and
            l2 > h1  # gap up
        )
        if bull:
            # Candle3 black
            if c3 < o3:
                # open3 inside body of candle2
                if min(o2, c2) < o3 < max(o2, c2):
                    # close3 inside the gap but not fully filling it
                    if h1 < c3 < l2:
                        out[i] = True
                        continue
        # --- Bearish Tasuki Gap ---
        # Candle1 black, Candle2 black with gap down
        bear = (
            c1 < o1 and
            c2 < o2 and
            h2 < l1  # gap down
        )
        if bear:
            # Candle3 white
            if c3 > o3:
                # open3 inside body of candle2
                if min(o2, c2) < o3 < max(o2, c2):
                    # close3 inside the gap but not fully filling it
                    if h2 < c3 < l1:
                        out[i] = True
                        continue
    return out


def cdl_tasukigap(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Tasuki Gap pattern.
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
        talib_out = talib.CDLTASUKIGAP(open_, high, low, close)
        talib_out = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(talib_out, offset, fillna)
    mask = _cdl_tasukigap_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_tasukigap_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_TASUKIGAP",
) -> pl.DataFrame:
    """
    Add Tasuki Gap column to Polars DataFrame.
    """
    out = cdl_tasukigap(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
