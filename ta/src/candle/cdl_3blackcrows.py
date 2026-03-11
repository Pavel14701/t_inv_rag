# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import njit, types

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (types.float64[:], types.float64[:], types.float64[:], types.float64[:]),
    cache=True,
    fastmath=True
)
def _cdl_3blackcrows_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Three Black Crows pattern.
    Returns float64 mask: 1.0 where pattern completes, else 0.0.
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)
    for i in range(2, n):
        # --- candle 1 ---
        o2 = open_[i - 2]
        c2 = close[i - 2]
        if c2 >= o2:
            continue
        # --- candle 2 ---
        o1 = open_[i - 1]
        c1 = close[i - 1]
        if c1 >= o1:
            continue
        # open inside previous body
        if not (o1 < o2 and o1 > c2):
            continue
        # --- candle 3 ---
        o0 = open_[i]
        c0 = close[i]
        if c0 >= o0:
            continue
        if not (o0 < o1 and o0 > c1):
            continue
        # closes lower each time
        if not (c2 > c1 > c0):
            continue
        out[i] = 1.0
    return out


def cdl_3blackcrows(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Three Black Crows pattern.
    Returns numpy array of float64: 1.0 where pattern occurs, else 0.0.
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
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    # TA-Lib branch
    if use_talib and talib_available:
        talib_out = talib.CDL3BLACKCROWS(open_, high, low, close)
        # Convert to binary mask (TA-Lib returns 0 or -100)
        talib_out = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(talib_out, offset, fillna)

    # Numba branch
    out = _cdl_3blackcrows_nb(open_, high, low, close)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_3blackcrows_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_3BLACKCROWS",
) -> pl.DataFrame:
    """
    Add Three Black Crows column to Polars DataFrame.
    """
    out = cdl_3blackcrows(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))