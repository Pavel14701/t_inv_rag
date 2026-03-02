# -*- coding: utf-8 -*-
"""
Midprice indicator – Numba‑accelerated with TA‑Lib fallback.
"""

from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core Numba implementation (single pass for min of low and max of high)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _midprice_numba_core(high: np.ndarray, low: np.ndarray, length: int) -> np.ndarray:
    """
    Compute midprice = (rolling_min(low) + rolling_max(high)) / 2 in one pass.

    Parameters
    ----------
    high, low : np.ndarray
        Price arrays (float64), same length.
    length : int
        Window length.

    Returns
    -------
    np.ndarray
        Midprice values; first `length-1` positions are NaN.
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        # Initialize with the first value in the window
        mn = low[i - length + 1]
        mx = high[i - length + 1]
        # Scan the window
        for j in range(i - length + 2, i + 1):
            _low = low[j]
            _high = high[j]
            if _low < mn:
                mn = _low
            if _high > mx:
                mx = _high
        out[i] = (mn + mx) * 0.5
    return out


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def midprice_numba(
    high: np.ndarray,
    low: np.ndarray,
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Midprice using Numba (raw numpy version).
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)

    result = _midprice_numba_core(high, low, length)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# TA‑Lib wrapper
# ----------------------------------------------------------------------
def midprice_talib(
    high: np.ndarray,
    low: np.ndarray,
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Midprice using TA‑Lib.
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")

    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)

    result = talib.MIDPRICE(high, low, timeperiod=length)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def midprice_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal Midprice with backend selection.

    Parameters
    ----------
    high, low : np.ndarray or pl.Series
        High and low price series.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA‑Lib if available.

    Returns
    -------
    np.ndarray
        Midprice values.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()

    if use_talib and talib_available:
        return midprice_talib(high, low, length, offset, fillna)
    else:
        return midprice_numba(high, low, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def midprice_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Add Midprice column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col : str
        Columns with high and low prices.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA‑Lib if available.
    output_col : str, optional
        Output column name (default f"MIDPRICE_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    result = midprice_ind(high, low, length, offset, fillna, use_talib)
    out_name = output_col or f"MIDPRICE_{length}"
    return df.with_columns([pl.Series(out_name, result)])