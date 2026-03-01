# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Fibonacci Weighted Moving Average (FWMA) – Numba implementation with caching
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _get_fib_weights(length: int, asc: bool) -> np.ndarray:
    """
    Generate normalized Fibonacci weights for given length and direction.
    Results are cached.
    """
    w = np.zeros(length, dtype=np.float64)
    w[0] = 1.0
    if length > 1:
        w[1] = 1.0
        for i in range(2, length):
            w[i] = w[i - 1] + w[i - 2]
    if not asc:
        w = w[::-1]
    w /= w.sum()
    return w


@jit(nopython=True, fastmath=True, cache=True)
def _fwma_numba_cached(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    FWMA core loop with precomputed weights (Numba).

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array.
    weights : np.ndarray
        Normalized weights (length = window size).

    Returns
    -------
    np.ndarray
        FWMA values; first (len(weights)-1) positions are NaN.
    """
    n = len(arr)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        # weights correspond to arr[i - (length-1) .. i] in direct order
        for j in range(length):
            acc += arr[i - j] * weights[length - 1 - j]
        out[i] = acc
    return out


def fwma_numba(
    close: np.ndarray,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Fibonacci Weighted Moving Average using Numba (with cached weights).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        FWMA period.
    asc : bool
        If True, recent values have higher weight.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        FWMA values.
    """
    close = close.astype(np.float64)
    weights = _get_fib_weights(length, asc)          # from cache or compute
    fwma = _fwma_numba_cached(close, weights)
    return _apply_offset_fillna(fwma, offset, fillna)


def fwma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal FWMA (always uses Numba, no TA-Lib equivalent).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        FWMA period.
    asc : bool
        If True, recent values have higher weight.
    offset : int
        Shift result.
    fillna : float, optional
        Fill NaN with this value.

    Returns
    -------
    np.ndarray
        FWMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return fwma_numba(close, length, asc, offset, fillna)


def fwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    FWMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        FWMA period.
    asc : bool
        If True, recent values have higher weight.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"FWMA_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.
    """
    close = df[close_col].to_numpy()
    result = fwma_ind(
        close,
        length=length,
        asc=asc,
        offset=offset,
        fillna=fillna
    )
    out_name = output_col or f"FWMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])