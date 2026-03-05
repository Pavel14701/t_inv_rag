# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _mad_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """
    Rolling Mean Absolute Deviation with sliding window and online mean update.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.

    Returns
    -------
    np.ndarray
        MAD values; first `length-1` positions are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    s = 0.0
    for i in range(length):
        s += close[i]
    mean = s / length
    mad = 0.0
    for i in range(length):
        mad += abs(close[i] - mean)
    mad /= length
    out[length - 1] = mad
    for i in range(length, n):
        # Update mean using the removed and added values
        old = close[i - length]
        new = close[i]
        s += new - old
        mean = s / length
        # Compute MAD for the new window
        mad = 0.0
        for j in range(i - length + 1, i + 1):
            mad += abs(close[j] - mean)
        mad /= length
        out[i] = mad
    return out


def mad_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Rolling Mean Absolute Deviation using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    result = _mad_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def mad_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal rolling MAD (always uses Numba).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        MAD values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return mad_numba(close, length, offset, fillna)


def mad_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """
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
        Output column name (default f"MAD_{length}").

    Returns
    -------
    pl.Series
    """
    close = df[close_col].to_numpy()
    result = mad_ind(close, length, offset, fillna)
    out_name = output_col or f"MAD_{length}"
    return pl.Series(out_name, result)