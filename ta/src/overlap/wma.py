# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Cached weights for WMA (linear weights)
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _get_wma_weights(length: int, asc: bool) -> np.ndarray:
    """
    Generate normalized linear weights for WMA.
    If asc=True, weights increase from 1 to length (most recent heaviest).
    If asc=False, weights decrease (most recent lightest).
    Weights are normalized so that sum = 1.
    """
    w = np.arange(1, length + 1, dtype=np.float64)
    if not asc:
        w = w[::-1]
    w /= w.sum()
    return w


# ----------------------------------------------------------------------
# WMA core loop (Numba)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _wma_numba_core(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted Moving Average core loop.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array.
    weights : np.ndarray
        Normalized weights (length = window size).

    Returns
    -------
    np.ndarray
        WMA values; first (len(weights)-1) positions are NaN.
    """
    n = len(arr)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        # weighted sum over the window
        for j in range(length):
            acc += arr[i - j] * weights[length - 1 - j]
        out[i] = acc
    return out


# ----------------------------------------------------------------------
# WMA using Numba (with offset and fillna)
# ----------------------------------------------------------------------
def wma_numba(
    close: np.ndarray,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Weighted Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        WMA period.
    asc : bool
        If True, recent values have higher weight (default).
        If False, older values have higher weight.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        WMA values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    weights = _get_wma_weights(length, asc)
    wma = _wma_numba_core(close, weights)
    return _apply_offset_fillna(wma, offset, fillna)


# ----------------------------------------------------------------------
# WMA via TA-Lib (only asc=True)
# ----------------------------------------------------------------------
def wma_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Weighted Moving Average via TA-Lib (asc=True only).
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    wma = talib.WMA(close, timeperiod=length)
    from ..utils import _apply_offset_fillna
    return _apply_offset_fillna(wma, offset, fillna)


# ----------------------------------------------------------------------
# Universal WMA function
# ----------------------------------------------------------------------
def wma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal Weighted Moving Average with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        WMA period.
    asc : bool
        If True, recent values have higher weight (TA-Lib compatible).
        If False, older values have higher weight (Numba only).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        If True and TA-Lib is available, use it (only for asc=True).
        If asc=False, falls back to Numba regardless of this flag.

    Returns
    -------
    np.ndarray
        WMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available and asc:
        return wma_talib(close, length, offset, fillna)
    else:
        return wma_numba(close, length, asc, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def wma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    WMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        WMA period.
    asc : bool
        If True, recent values have higher weight.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA-Lib if available and asc=True.
    output_col : str, optional
        Output column name (default f"WMA_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.
    """
    close = df[close_col].to_numpy()
    result = wma_ind(
        close,
        length=length,
        asc=asc,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib
    )
    out_name = output_col or f"WMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])