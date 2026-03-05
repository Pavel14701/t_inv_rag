# -*- coding: utf-8 -*-
"""
Ehlers Super Smoother Filter (SSF) – aggressively optimized Numba version.
"""

from typing import Optional

import numpy as np
import polars as pl
from numba import njit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Original Ehlers SSF (2‑pole) with fastmath
# ----------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def _ssf_ehlers(x: np.ndarray, n: int, pi: float, sqrt2: float) -> np.ndarray:
    m = len(x)
    ratio = sqrt2 / n
    a = np.exp(-pi * ratio)
    b = 2.0 * a * np.cos(180.0 * ratio)   # degrees
    c = a * a - b + 1.0
    out = np.empty(m, dtype=np.float64)
    out[0] = x[0]
    out[1] = x[1]
    for i in range(2, m):
        out[i] = 0.5 * c * (
            x[i] + x[i - 1]
        ) + b * out[i - 1] - a * a * out[i - 2]
    return out


# ----------------------------------------------------------------------
# Everget's version (uses pi instead of 180) with fastmath
# ----------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def _ssf_everget(x: np.ndarray, n: int, pi: float, sqrt2: float) -> np.ndarray:
    m = len(x)
    arg = pi * sqrt2 / n
    a = np.exp(-arg)
    b = 2.0 * a * np.cos(arg)
    out = np.empty(m, dtype=np.float64)
    out[0] = x[0]
    out[1] = x[1]
    for i in range(2, m):
        out[i] = 0.5 * (
            a * a - b + 1.0
        ) * (x[i] + x[i - 1]) + b * out[i - 1] - a * a * out[i - 2]
    return out


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def ssf_numba(
    close: np.ndarray,
    length: int = 20,
    everget: bool = False,
    pi: float = 3.14159,
    sqrt2: float = 1.414,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Super Smoother Filter using Numba.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if everget:
        result = _ssf_everget(close, length, pi, sqrt2)
    else:
        result = _ssf_ehlers(close, length, pi, sqrt2)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def ssf_ind(
    close: np.ndarray | pl.Series,
    length: int = 20,
    everget: bool = False,
    pi: float = 3.14159,
    sqrt2: float = 1.414,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Universal SSF.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ssf_numba(close, length, everget, pi, sqrt2, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def ssf_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 20,
    everget: bool = False,
    pi: float = 3.14159,
    sqrt2: float = 1.414,
    offset: int = 0,
    fillna: Optional[float] = None,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Add SSF column to Polars DataFrame.
    """
    close = df[close_col].to_numpy()
    result = ssf_ind(close, length, everget, pi, sqrt2, offset, fillna)
    out_name = output_col or f"SSF{'e' if everget else ''}_{length}"
    return df.with_columns([pl.Series(out_name, result)])