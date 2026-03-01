# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit


# ----------------------------------------------------------------------
# Core Numba implementation of SMMA
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _smma_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """
    Smoothed Moving Average (SMMA) core calculation.

    First value (at index length-1) is SMA of first `length` elements.
    Then: SMMA[i] = ((length-1) * SMMA[i-1] + close[i]) / length
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial SMA
    s = 0.0
    for i in range(length):
        s += close[i]
    out[length - 1] = s / length
    # Recurrence
    for i in range(length, n):
        out[i] = ((length - 1) * out[i - 1] + close[i]) / length
    return out


# ----------------------------------------------------------------------
# SMMA using Numba (with offset and fillna)
# ----------------------------------------------------------------------
def smma_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Smoothed Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        SMMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        SMMA values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _smma_numba_core(close, length)
    if offset != 0:
        result = np.roll(result, offset)
        if offset > 0:
            result[:offset] = np.nan
        else:
            result[offset:] = np.nan
    if fillna is not None:
        result = np.where(np.isnan(result), fillna, result)
    return result


# ----------------------------------------------------------------------
# Universal SMMA function (always uses Numba)
# ----------------------------------------------------------------------
def smma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal Smoothed Moving Average (Numba only).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        SMMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        SMMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return smma_numba(close, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def smma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    SMMA for Polars DataFrame (Numba only).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        SMMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"SMMA_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.
    """
    close = df[close_col].to_numpy()
    result = smma_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna
    )
    out_name = output_col or f"SMMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])