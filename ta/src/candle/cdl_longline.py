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
def _cdl_longline_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Long Line Candle pattern.
    Returns boolean mask where pattern completes (True at the candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        o = open_[i]
        c = close[i]
        h = high[i]
        l = low[i]
        rng = h - l
        if rng <= 0.0:
            continue
        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l
        # Long body: тело должно быть значительной частью диапазона
        if body < 0.6 * rng:
            continue
        # Short shadows: обе тени должны быть маленькими
        if upper > 0.2 * rng:
            continue
        if lower > 0.2 * rng:
            continue
        out[i] = True
    return out


def cdl_longline(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Long Line Candle pattern.
    Returns numpy array of float64: 1.0 where pattern occurs, else 0.0.
    """
    # Polars → numpy
    if isinstance(open_, pl.Series): 
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series): 
        high = high.to_numpy()
    if isinstance(low, pl.Series): 
        low = low.to_numpy()
    if isinstance(close, pl.Series): 
        close = close.to_numpy()
    # Ensure float64 + contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # TA‑Lib branch
    if use_talib and talib_available:
        talib_out = talib.CDLLONGLINE(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    # Numba branch
    mask = _cdl_longline_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_longline_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_LONGLINE",
) -> pl.DataFrame:
    """
    Add Long Line Candle column to Polars DataFrame.
    """
    out = cdl_longline(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
