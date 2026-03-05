# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Cached symmetric triangle weights
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _symmetric_weights(length: int) -> np.ndarray:
    """
    Generate normalized symmetric triangle weights.
    For length n, weights form a symmetric triangle: [1,2,...,2,1] (or [1,2,...,2,1]).
    Normalized so sum = 1.
    """
    if length % 2 == 0:
        # even: e.g. n=4 -> [1,2,2,1]
        half = length // 2
        w = np.concatenate([np.arange(1, half + 1), np.arange(half, 0, -1)])
    else:
        # odd: e.g. n=5 -> [1,2,3,2,1]
        half = length // 2
        w = np.concatenate([np.arange(1, half + 2), np.arange(half, 0, -1)])
    w = w.astype(np.float64)
    w /= w.sum()
    return w


# ----------------------------------------------------------------------
# Core SWMA calculation in Numba (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _swma_numba_core(close: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    SWMA core loop.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    weights : np.ndarray
        Normalized symmetric weights (length = window size).

    Returns
    -------
    np.ndarray
        SWMA values; first (len(weights)-1) positions are NaN.
    """
    n = len(close)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += close[i - j] * weights[length - 1 - j]
        out[i] = acc
    return out


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def swma_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    SWMA using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    weights = _symmetric_weights(length)
    result = _swma_numba_core(close, weights)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def swma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal SWMA (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return swma_numba(close, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def swma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    Add SWMA column to Polars DataFrame.

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
        Output column name (default f"SWMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with SWMA column.
    """
    close = df[close_col].to_numpy()
    result = swma_ind(close, length, offset, fillna)
    out_name = output_col or f"SWMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])