# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core MCGD calculation in Numba (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _mcgd_numba_core(close: np.ndarray, length: int, c: float) -> np.ndarray:
    """
    McGinley Dynamic core loop.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Period parameter (n in the formula).
    c : float
        Denominator multiplier (usually 1, sometimes 0.6).

    Returns
    -------
    np.ndarray
        MCGD values; first element equals first close price.
    """
    n = len(close)
    mcgd = np.empty(n, dtype=np.float64)
    mcgd[0] = close[0]

    for i in range(1, n):
        # MCGD formula: MCGD[i] = MCGD[i-1] + (close[i] - MCGD[i-1]) / (
        # c * n * (close[i] / MCGD[i-1])**4)
        # Avoid division by zero if MCGD[i-1] == 0 (should not happen with prices)
        ratio = close[i] / mcgd[i - 1]
        denom = c * length * (ratio ** 4)
        # If denom is too large, the adjustment becomes tiny – safe.
        mcgd[i] = mcgd[i - 1] + (close[i] - mcgd[i - 1]) / denom
    return mcgd


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def mcgd_numba(
    close: np.ndarray,
    length: int = 10,
    c: float = 1.0,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    McGinley Dynamic using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Period parameter.
    c : float
        Denominator multiplier (0 < c ≤ 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        MCGD values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    mcgd = _mcgd_numba_core(close, length, c)
    return _apply_offset_fillna(mcgd, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def mcgd_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    c: float = 1.0,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal McGinley Dynamic (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return mcgd_numba(close, length, c, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def mcgd_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    c: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    Add MCGD column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length : int
        Period parameter.
    c : float
        Denominator multiplier (0 < c ≤ 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"MCGD_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with MCGD series.
    """
    close = df[close_col].to_numpy()
    result = mcgd_ind(close, length, c, offset, fillna)
    out_name = output_col or f"MCGD_{length}"
    return df.with_columns([pl.Series(out_name, result)])