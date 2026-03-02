# -*- coding: utf-8 -*-
"""
Midpoint indicator – Numba‑accelerated with TA‑Lib fallback.
"""

from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core Numba implementation (single pass for min and max)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _midpoint_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """
    Compute midpoint = (rolling_min + rolling_max) / 2 in one pass.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # For each window, compute min and max
    for i in range(length - 1, n):
        mn = close[i - length + 1]
        mx = mn
        for j in range(i - length + 2, i + 1):
            val = close[j]
            if val < mn:
                mn = val
            if val > mx:
                mx = val
        out[i] = (mn + mx) * 0.5
    return out


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def midpoint_numba(
    close: np.ndarray,
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Midpoint using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _midpoint_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# TA‑Lib wrapper
# ----------------------------------------------------------------------
def midpoint_talib(
    close: np.ndarray,
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Midpoint using TA‑Lib.
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = talib.MIDPOINT(close, timeperiod=length)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def midpoint_ind(
    close: np.ndarray | pl.Series,
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal Midpoint with backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
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
        Midpoint values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return midpoint_talib(close, length, offset, fillna)
    else:
        return midpoint_numba(close, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def midpoint_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 2,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Add Midpoint column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA‑Lib if available.
    output_col : str, optional
        Output column name (default f"MIDPOINT_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    result = midpoint_ind(close, length, offset, fillna, use_talib)
    out_name = output_col or f"MIDPOINT_{length}"
    return df.with_columns([pl.Series(out_name, result)])