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
def _cdl_longleggeddoji_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Long-Legged Doji pattern.
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
        # Body
        body = c - o if c > o else o - c
        # Doji: очень маленькое тело
        if body > 0.1 * rng:
            continue
        # Shadows
        upper = h - (c if c > o else o)
        lower = (c if c > o else o) - l
        # Long-legged: обе тени длинные
        if upper < 0.4 * rng:
            continue
        if lower < 0.4 * rng:
            continue
        out[i] = True
    return out


def cdl_longleggeddoji(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Long-Legged Doji pattern.
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
        talib_out = talib.CDLLONGLEGGEDDOJI(open_, high, low, close)
        talib_out = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(talib_out, offset, fillna)
    mask = _cdl_longleggeddoji_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_longleggeddoji_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_LONGLEGGEDDOJI",
) -> pl.DataFrame:
    """
    Add Long-Legged Doji column to Polars DataFrame.
    """
    out = cdl_longleggeddoji(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
