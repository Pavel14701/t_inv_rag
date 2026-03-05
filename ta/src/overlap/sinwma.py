# -*- coding: utf-8 -*-
from functools import lru_cache
from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Cached sine weights
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _sine_weights(length: int) -> np.ndarray:
    """
    Generate normalized sine weights for SINWMA.
    Formula: w_i = sin((i+1) * pi / (length+1)), then normalized.
    """
    i = np.arange(1, length + 1, dtype=np.float64)
    w = np.sin(i * np.pi / (length + 1))
    w /= w.sum()
    return w


# ----------------------------------------------------------------------
# Core SINWMA calculation in Numba (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _sinwma_numba_core(close: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    SINWMA core loop.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    weights : np.ndarray
        Normalized sine weights (length = window size).

    Returns
    -------
    np.ndarray
        SINWMA values; first (len(weights)-1) positions are NaN.
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
def sinwma_numba(
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    SINWMA using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    weights = _sine_weights(length)
    result = _sinwma_numba_core(close, weights)

    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def sinwma_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Universal SINWMA (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return sinwma_numba(close, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def sinwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    offset: int = 0,
    fillna: Optional[float] = None,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Add SINWMA column to Polars DataFrame.

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
        Output column name (default f"SINWMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with SINWMA column.
    """
    close = df[close_col].to_numpy()
    result = sinwma_ind(close, length, offset, fillna)
    out_name = output_col or f"SINWMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])