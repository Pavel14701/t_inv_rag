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
def _cdl_morningdojistar_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Morning Doji Star pattern.
    Returns boolean mask where pattern completes (True at the 3rd candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # --- Candle 1: long black ---
        o1 = open_[i - 2]
        c1 = close[i - 2]
        h1 = high[i - 2]
        l1 = low[i - 2]
        if not (c1 < o1):
            continue
        rng1 = h1 - l1
        body1 = o1 - c1
        if rng1 <= 0.0 or body1 < 0.6 * rng1:
            continue  # must be long bearish candle
        # --- Candle 2: doji with gap down ---
        o2 = open_[i - 1]
        c2 = close[i - 1]
        h2 = high[i - 1]
        l2 = low[i - 1]
        rng2 = h2 - l2
        if rng2 <= 0.0:
            continue
        body2 = abs(c2 - o2)
        if body2 > 0.1 * rng2:
            continue  # must be doji
        # gap down: doji entirely below candle 1 body
        if not (h2 < c1):
            continue
        # --- Candle 3: bullish reversal ---
        o3 = open_[i]
        c3 = close[i]
        if not (c3 > o3):
            continue
        # must close at least above midpoint of candle 1 body
        midpoint1 = (o1 + c1) * 0.5
        if not (c3 > midpoint1):
            continue
        out[i] = True
    return out


def cdl_morningdojistar(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Morning Doji Star pattern.
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
        talib_out = talib.CDLMORNINGDOJISTAR(open_, high, low, close)
        # TA‑Lib returns +100 → convert to binary mask
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    # Numba branch
    mask = _cdl_morningdojistar_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_morningdojistar_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_MORNINGDOJISTAR",
) -> pl.DataFrame:
    """
    Add Morning Doji Star column to Polars DataFrame.
    """
    out = cdl_morningdojistar(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
