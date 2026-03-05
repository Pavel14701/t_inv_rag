# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# WCP using Numba (vectorised)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _wcp_numba_core(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Weighted Closing Price core: (high + low + 2*close) / 4
    """
    return (high + low + 2.0 * close) * 0.25


def wcp_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    WCP using Numba (raw numpy version).
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Ensure contiguous (though not strictly needed for simple arithmetic)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    result = _wcp_numba_core(high, low, close)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# WCP using TA‑Lib (if available)
# ----------------------------------------------------------------------
def wcp_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    WCP using TA‑Lib (C implementation). Returns (high + low + 2*close)/4.
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    result = talib.WCLPRICE(high, low, close)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def wcp_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal Weighted Closing Price with backend selection.

    Parameters
    ----------
    high, low, close : np.ndarray or pl.Series
        Price series.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        If True and TA‑Lib is available, use it; else use Numba.

    Returns
    -------
    np.ndarray
        WCP values.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return wcp_talib(high, low, close, offset, fillna)
    else:
        return wcp_numba(high, low, close, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def wcp_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str = "WCP"
) -> pl.DataFrame:
    """
    Add WCP column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Column names for price components.
    offset, fillna, use_talib : as above.
    output_col : str, optional
        Name of the output column (default "WCP").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with WCP column.
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = wcp_ind(high, low, close, offset, fillna, use_talib)
    return df.with_columns([pl.Series(output_col, result)])