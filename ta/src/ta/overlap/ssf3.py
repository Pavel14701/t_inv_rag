# -*- coding: utf-8 -*-
"""
Ehlers 3‑Pole Super Smoother Filter (SSF3) – Numba‑accelerated with Polars integration.
"""

from typing import Optional

import numpy as np
import polars as pl
from numba import njit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core SSF3 calculation (Numba)
# ----------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def _ssf3_numba_core(
    close: np.ndarray,
    length: int,
    pi: float,
    sqrt3: float
) -> np.ndarray:
    """
    John F. Ehlers' 3‑pole Super Smoother Filter (Everget variant).
    """
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    # First three values are just the input (no filtering yet)
    out[0] = close[0]
    if n > 1:
        out[1] = close[1]
    if n > 2:
        out[2] = close[2]
    if n < 4:
        return out
    a = np.exp(-pi / length)
    b = 2.0 * a * np.cos(-pi * sqrt3 / length)
    c = a * a
    d4 = c * c
    d3 = -c * (1.0 + b)
    d2 = b + c
    d1 = 1.0 - d2 - d3 - d4
    for i in range(3, n):
        out[i] = d1 * close[i] + d2 * \
            out[i - 1] + d3 * out[i - 2] + d4 * out[i - 3]
    return out


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def ssf3_numba(
    close: np.ndarray,
    length: int = 20,
    pi: float = 3.14159,
    sqrt3: float = 1.732,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    3‑pole Super Smoother Filter using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Filter period.
    pi : float
        Value of π (default 3.14159).
    sqrt3 : float
        Value of √3 (default 1.732).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        SSF3 values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    result = _ssf3_numba_core(close, length, pi, sqrt3)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def ssf3_ind(
    close: np.ndarray | pl.Series,
    length: int = 20,
    pi: float = 3.14159,
    sqrt3: float = 1.732,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Universal 3‑pole Super Smoother Filter (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ssf3_numba(close, length, pi, sqrt3, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def ssf3_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 20,
    pi: float = 3.14159,
    sqrt3: float = 1.732,
    offset: int = 0,
    fillna: Optional[float] = None,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Add SSF3 column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, pi, sqrt3, offset, fillna : as above.
    output_col : str, optional
        Output column name (default f"SSF3_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with SSF3 column.
    """
    close = df[close_col].to_numpy()
    result = ssf3_ind(close, length, pi, sqrt3, offset, fillna)
    out_name = output_col or f"SSF3_{length}"
    return df.with_columns([pl.Series(out_name, result)])