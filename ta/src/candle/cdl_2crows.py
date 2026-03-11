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
def _cdl_2crows_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Two Crows pattern.
    Returns boolean mask where pattern completes (True at the third candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # First candle (i-2) must be white (bullish)
        if close[i - 2] <= open_[i - 2]:
            continue
        # Second candle (i-1) must be black (bearish)
        if close[i - 1] >= open_[i - 1]:
            continue
        # Second candle opens above first close
        if not (open_[i - 1] > close[i - 2]):
            continue
        # Second candle closes above first open
        if not (close[i - 1] > open_[i - 2]):
            continue
        # Second candle closes below first close
        if not (close[i - 1] < close[i - 2]):
            continue
        # Third candle must be black
        if close[i] >= open_[i]:
            continue
        # Third candle opens above second close
        if not (open_[i] > close[i - 1]):
            continue
        # Third candle closes below second close
        if not (close[i] < close[i - 1]):
            continue
        out[i] = True
    return out


def cdl_2crows(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Two Crows pattern.
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
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

    # TA‑Lib branch
    if use_talib and talib_available:
        talib_out = talib.CDL2CROWS(open_, high, low, close)
        # Convert to binary mask (TA‑Lib returns 0 or -100)
        talib_out = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(talib_out, offset, fillna)

    # Numba branch
    mask = _cdl_2crows_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_2crows_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_2CROWS",
) -> pl.DataFrame:
    """
    Add Two Crows column to Polars DataFrame.
    """
    out = cdl_2crows(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))