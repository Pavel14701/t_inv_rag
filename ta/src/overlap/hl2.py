# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Optimized HL2(сдвиг и fillna через общую утилиту)
# ----------------------------------------------------------------------
def _hl2(
    high: np.ndarray,
    low: np.ndarray,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    HL2 (average of high and low) using Numba (optimized).

    Parameters
    ----------
    high, low : np.ndarray
        Price arrays (float64).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        HL2 values with applied offset and fillna.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    avg = (high + low) * 0.5
    return _apply_offset_fillna(avg, offset, fillna)


# ----------------------------------------------------------------------
# Универсальная функция HL2 (принимает np.ndarray или pl.Series)
# ----------------------------------------------------------------------
def hl2_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal HL2 (always uses Numba).
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    return _hl2(high, low, offset, fillna)


# ----------------------------------------------------------------------
# Интеграция с Polars DataFrame
# ----------------------------------------------------------------------
def hl2_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    HL2 for Polars DataFrame.

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
        Output column name (default "HL2").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.    
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    result = hl2_ind(high, low, offset, fillna)
    out_name = output_col or "HL2"
    return df.with_columns([pl.Series(out_name, result)])