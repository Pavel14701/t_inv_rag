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
def _cdl_mathold_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Mat Hold pattern.
    Returns boolean mask where pattern completes (True at the 5th candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(4, n):
        # --- Candle 1: strong bullish ---
        o1 = open_[i - 4]
        c1 = close[i - 4]
        h1 = high[i - 4]
        l1 = low[i - 4]
        if not (c1 > o1):
            continue
        rng1 = h1 - l1
        body1 = c1 - o1
        if rng1 <= 0.0 or body1 < 0.6 * rng1:
            continue  # must be long bullish candle
        # --- Candles 2–4: small pullback candles ---
        pullback_ok = True
        for k in range(3, 0, -1):  # i-3, i-2, i-1
            o = open_[i - k]
            c = close[i - k]
            h = high[i - k]
            l = low[i - k]
            rng = h - l
            body = abs(c - o)
            if rng <= 0.0:
                pullback_ok = False
                break
            # small bodies
            if body > 0.4 * rng1:
                pullback_ok = False
                break
            # must stay within body of candle 1
            if h > c1 or l < o1:
                pullback_ok = False
                break
        if not pullback_ok:
            continue
        # --- Candle 5: bullish breakout ---
        o5 = open_[i]
        c5 = close[i]
        if not (c5 > o5):
            continue
        # must close above candle 1 close
        if not (c5 > c1):
            continue
        out[i] = True
    return out


def cdl_mathold(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Mat Hold pattern.
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
        talib_out = talib.CDLMATHOLD(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    # Numba branch
    mask = _cdl_mathold_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_mathold_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_MATHOLD",
) -> pl.DataFrame:
    """
    Add Mat Hold column to Polars DataFrame.
    """
    out = cdl_mathold(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
