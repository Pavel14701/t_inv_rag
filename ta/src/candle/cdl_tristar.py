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
def _cdl_tristar_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Tristar pattern.
    Returns boolean mask where pattern completes (True at the 3rd doji).
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
        h3 = high[i]
        l3 = low[i]
        # must be doji: body extremely small
        rng1 = h1 - l1
        rng2 = h2 - l2
        rng3 = h3 - l3
        if rng1 <= 0 or rng2 <= 0 or rng3 <= 0:
            continue
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        if body1 > 0.1 * rng1:
            continue
        if body2 > 0.1 * rng2:
            continue
        if body3 > 0.1 * rng3:
            continue
        # gaps between doji
        # gap between candle1 and candle2
        gap12 = (l2 > h1) or (h2 < l1)
        if not gap12:
            continue
        # gap between candle2 and candle3
        gap23 = (l3 > h2) or (h3 < l2)
        if not gap23:
            continue
        # bullish or bearish tristar — direction irrelevant for binary output
        out[i] = True
    return out


def cdl_tristar(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Tristar pattern.
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
        talib_out = talib.CDLTRISTAR(open_, high, low, close)
        talib_out = (talib_out != 0).astype(np.float64)  # +100 / -100 → 1.0
        return _apply_offset_fillna(talib_out, offset, fillna)
    mask = _cdl_tristar_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_tristar_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_TRISTAR",
) -> pl.DataFrame:
    """
    Add Tristar pattern column to Polars DataFrame.
    """
    out = cdl_tristar(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))