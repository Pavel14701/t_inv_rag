# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Optimized HL3(сдвиг и fillna через общую утилиту)
# ----------------------------------------------------------------------
def _hlc3(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    HLC3 (average of high, low and close) optimized.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        HLC3 values with applied offset and fillna.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    avg = (high + low + close) / 3.0
    return _apply_offset_fillna(avg, offset, fillna)


# ----------------------------------------------------------------------
# Универсальная функция HL3 (принимает np.ndarray или pl.Series)
# ----------------------------------------------------------------------
def hlc3_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal HLC3.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return _hlc3(high, low, close, offset, fillna)


# ----------------------------------------------------------------------
# Интеграция с Polars DataFrame
# ----------------------------------------------------------------------
def hlc3_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    HLC3 for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col, low_col : str
        Names of the columns with high and low prices.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default "HLC3").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.    
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = hlc3_ind(high, low, close, offset, fillna)
    out_name = output_col or "HLC3"
    return df.with_columns([pl.Series(out_name, result)])