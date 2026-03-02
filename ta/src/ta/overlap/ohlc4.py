# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import polars as pl

from ..utils import _apply_offset_fillna


def ohlc4_numpy(
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    OHLC4 using pure NumPy (vectorised, no JIT needed).

    Parameters
    ----------
    open, high, low, close : np.ndarray
        Price arrays (float64).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        OHLC4 values.
    """
    # Ensure float64 and C‑contiguous with minimal copying
    open = np.asarray(open, dtype=np.float64, copy=False)
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Compute average (vectorised)
    avg = (open + high + low + close) * 0.25
    # Apply offset and fillna
    return _apply_offset_fillna(avg, offset, fillna)


def ohlc4_ind(
    open: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Universal OHLC4 (always uses NumPy).

    Parameters
    ----------
    open, high, low, close : np.ndarray or pl.Series
        Price series.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        OHLC4 values.
    """
    if isinstance(open, pl.Series):
        open = open.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ohlc4_numpy(open, high, low, close, offset, fillna)


def ohlc4_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: Optional[float] = None,
    output_col: str = "OHLC4"
) -> pl.DataFrame:
    """
    Add OHLC4 column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    open_col, high_col, low_col, close_col : str
        Column names for price components.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Name of the output column (default "OHLC4").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with OHLC4 column.
    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    result = ohlc4_numpy(open_arr, high_arr, low_arr, close_arr, offset, fillna)
    return df.with_columns([pl.Series(output_col, result)])