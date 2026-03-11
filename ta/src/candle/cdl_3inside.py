# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import njit, types

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (types.float64[:], types.float64[:], types.float64[:], types.float64[:]),
    nopython=True,
    cache=True,
    fastmath=True
)
def _cdl_3inside_nb(
    open_: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Three Inside pattern.
    Returns float64 array:
        1.0  → bullish Three Inside Up
       -1.0  → bearish Three Inside Down
        0.0  → no pattern
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)
    for i in range(2, n):
        # Candle 1
        o2 = open_[i - 2]
        c2 = close[i - 2]
        # Candle 2
        o1 = open_[i - 1]
        c1 = close[i - 1]
        # Candle 3
        o0 = open_[i]
        c0 = close[i]
        # Bullish Three Inside Up
        if c2 < o2:  # 1st bearish
            if c1 > o1 and o1 < o2 and c1 > c2:  # 2nd bullish inside 1st
                if c0 > o0 and c0 > c1:  # 3rd bullish above 2nd close
                    out[i] = 1.0
                    continue
        # Bearish Three Inside Down
        if c2 > o2:  # 1st bullish
            if c1 < o1 and o1 > c2 and c1 < o2:  # 2nd bearish inside 1st
                if c0 < o0 and c0 < c1:  # 3rd bearish below 2nd close
                    out[i] = -1.0
    return out


def cdl_3inside(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Three Inside pattern.
    Returns numpy array of float64: 1.0 (bullish), -1.0 (bearish), 0.0 (none).
    """
    # Convert Polars Series to numpy
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    # Ensure float64 and contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # TA‑Lib branch
    if use_talib and talib_available:
        talib_out = talib.CDL3INSIDE(open_, high, low, close)
        # TA‑Lib returns 100, -100, or 0; convert to -1,0,1
        result = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(result, offset, fillna)
    # Numba branch
    out = _cdl_3inside_nb(open_, high, low, close)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_3inside_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_3INSIDE",
) -> pl.DataFrame:
    """
    Add Three Inside column to Polars DataFrame.
    """
    out = cdl_3inside(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))