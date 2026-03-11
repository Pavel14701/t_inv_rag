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
def _cdl_ladderbottom_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Ladder Bottom pattern.
    Returns boolean mask where pattern completes (True at the 5th candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(4, n):
        # --- Candle 1,2,3: three consecutive black candles ---
        ok = True
        for k in range(4, 1, -1):  # i-4, i-3, i-2
            if not (close[i - k] < open_[i - k]):
                ok = False
                break
        if not ok:
            continue
        # --- Candle 4: black with long lower shadow ---
        o4 = open_[i - 1]
        c4 = close[i - 1]
        h4 = high[i - 1]
        l4 = low[i - 1]
        if not (c4 < o4):  # black
            continue
        rng4 = h4 - l4
        if rng4 <= 0.0:
            continue
        lower4 = c4 - l4
        body4 = o4 - c4
        # long lower shadow
        if lower4 < 2.0 * body4:
            continue
        # --- Candle 5: bullish reversal ---
        o5 = open_[i]
        c5 = close[i]
        h5 = high[i]
        l5 = low[i]
        if not (c5 > o5):  # must be bullish
            continue
        # must close above body of candle 4
        if not (c5 > o4):
            continue
        out[i] = True
    return out


def cdl_ladderbottom(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Ladder Bottom pattern.
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
        talib_out = talib.CDLLADDERBOTTOM(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    # Numba branch
    mask = _cdl_ladderbottom_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_ladderbottom_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_LADDERBOTTOM",
) -> pl.DataFrame:
    """
    Add Ladder Bottom column to Polars DataFrame.
    """
    out = cdl_ladderbottom(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
