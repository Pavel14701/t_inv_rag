# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Cached weights for ALMA (avoid recomputation)
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _alma_weights(length: int, sigma: float, dist_offset: float) -> np.ndarray:
    """
    Generate normalized weights for ALMA.

    Parameters
    ----------
    length : int
        Window length.
    sigma : float
        Smoothing factor.
    dist_offset : float
        Distribution offset (0 to 1).

    Returns
    -------
    np.ndarray
        Normalized weights.
    """
    x = np.arange(length, dtype=np.float64)
    k = dist_offset * (length - 1)
    w = np.exp(-0.5 * ((sigma / length) * (x - k)) ** 2)
    w /= w.sum()
    return w


@jit(nopython=True, fastmath=True, cache=True)
def _alma_numba_full(
    arr: np.ndarray,
    weights: np.ndarray,
    offset: int,
    fillna: float | None
) -> np.ndarray:
    """
    ALMA core with integrated offset and fillna.
    """
    n = len(arr)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        # Not enough data – fill with fillna if provided
        if fillna is not None:
            out[:] = fillna
        return out
    # Main ALMA calculation
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += arr[i - j] * weights[length - 1 - j]
        out[i] = acc
    # Use the universal offset/fillna utility
    return _apply_offset_fillna(out, offset, fillna)


def alma_numba_opt(
    close: np.ndarray,
    length: int = 9,
    sigma: float = 6.0,
    dist_offset: float = 0.85,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Arnaud Legoux Moving Average using Numba (optimized).

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        ALMA period.
    sigma : float
        Smoothing factor.
    dist_offset : float
        Distribution offset (0 to 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        ALMA values.
    """
    # Minimize copying
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Ensure C-contiguous for best performance
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    weights = _alma_weights(length, sigma, dist_offset)
    return _alma_numba_full(close, weights, offset, fillna)


# ----------------------------------------------------------------------
# Universal ALMA function (TA-Lib not available)
# ----------------------------------------------------------------------
def alma_ind(
    close: np.ndarray | pl.Series,
    length: int = 9,
    sigma: float = 6.0,
    dist_offset: float = 0.85,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal ALMA (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return alma_numba_opt(close, length, sigma, dist_offset, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def alma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 9,
    sigma: float = 6.0,
    dist_offset: float = 0.85,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    ALMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        ALMA period.
    sigma : float
        Smoothing factor.
    dist_offset : float
        Distribution offset (0 to 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"ALMA_{length}_{sigma}_{dist_offset}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.    
    """
    close = df[close_col].to_numpy()
    result = alma_ind(
        close,
        length=length,
        sigma=sigma,
        dist_offset=dist_offset,
        offset=offset,
        fillna=fillna
    )
    out_name = output_col or f"ALMA_{length}_{sigma}_{dist_offset}"
    return df.with_columns([pl.Series(out_name, result)])