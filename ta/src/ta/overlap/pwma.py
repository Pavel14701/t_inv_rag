# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Cached Pascal's triangle weights (binomial coefficients)
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _pascal_weights(length: int, asc: bool) -> np.ndarray:
    """
    Generate normalized Pascal's triangle weights.
    Uses binomial coefficients from row (length-1) of Pascal's triangle.
    If asc=True, weights increase (most recent highest weight).
    """
    # Binomial coefficients for row (length-1)
    coeffs = np.zeros(length, dtype=np.float64)
    coeffs[0] = 1.0
    for k in range(1, length):
        coeffs[k] = coeffs[k - 1] * (length - k) / k
    if asc:
        w = coeffs
    else:
        w = coeffs[::-1]
    w /= w.sum()
    return w


# ----------------------------------------------------------------------
# Core PWMA calculation in Numba (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _pwma_numba_core(close: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    PWMA core loop.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    weights : np.ndarray
        Normalized weights (length = window size).

    Returns
    -------
    np.ndarray
        PWMA values; first (len(weights)-1) positions are NaN.
    """
    n = len(close)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        # weighted sum over the window, weights 
        # correspond to arr[i - (length-1) .. i] in direct order
        for j in range(length):
            acc += close[i - j] * weights[length - 1 - j]
        out[i] = acc
    return out


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def pwma_numba(
    close: np.ndarray,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    PWMA using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    weights = _pascal_weights(length, asc)
    result = _pwma_numba_core(close, weights)

    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def pwma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal PWMA (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return pwma_numba(close, length, asc, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def pwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    Add PWMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length : int
        Window length.
    asc : bool
        If True, most recent values have highest weight.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"PWMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with PWMA column.
    """
    close = df[close_col].to_numpy()
    result = pwma_ind(close, length, asc, offset, fillna)
    out_name = output_col or f"PWMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])