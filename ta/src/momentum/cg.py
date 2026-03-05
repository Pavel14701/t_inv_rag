# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _cg_numba_core(
    close: np.ndarray, 
    length: int
) -> np.ndarray:
    """
    Center of Gravity core with O(1) sliding window update.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.

    Returns
    -------
    np.ndarray
        CG values; first `length-1` positions are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # First window: compute directly
    numerator = 0.0
    denominator = 0.0
    for k in range(1, length + 1):
        val = close[k - 1]
        numerator += k * val
        denominator += val
    if denominator != 0.0:
        out[length - 1] = -numerator / denominator
    else:
        out[length - 1] = np.nan
    # Sliding window updates
    for i in range(length, n):
        oldest = close[i - length]      # price leaving the window
        newest = close[i]               # price entering the window
        # Update numerator and denominator
        numerator = numerator - oldest + length * newest
        denominator = denominator - oldest + newest
        if denominator != 0.0:
            out[i] = -numerator / denominator
        else:
            out[i] = np.nan
    return out


def cg_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Center of Gravity using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _cg_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def cg_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal Center of Gravity (always uses Numba).

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

    Returns
    -------
    np.ndarray
        CG values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return cg_numba(close, length, offset, fillna)


def cg_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date", 
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add Center of Gravity column to Polars DataFrame.

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
    output_col : str, optional
        Output column name (default f"CG_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    result = cg_ind(close, length, offset, fillna)
    out_name = output_col or f"CG_{length}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })