# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..utils import _apply_offset_fillna
from . import sma_ind


# ----------------------------------------------------------------------
# TRIMA using Numba (double SMA)
# ----------------------------------------------------------------------
def trima_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Triangular Moving Average using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    half_length = (length + 1) // 2  # round(0.5*(length+1))
    sma1 = sma_ind(close, half_length, use_talib=False)
    trima = sma_ind(sma1, half_length, use_talib=False)
    return _apply_offset_fillna(trima, offset, fillna)


# ----------------------------------------------------------------------
# TRIMA using TA‑Lib (if available)
# ----------------------------------------------------------------------
def trima_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Triangular Moving Average using TA‑Lib (C implementation).
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    trima = talib.TRIMA(close, timeperiod=length)
    return _apply_offset_fillna(trima, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def trima_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal TRIMA with backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        If True and TA‑Lib is available, use it; else use Numba.

    Returns
    -------
    np.ndarray
        TRIMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return trima_talib(close, length, offset, fillna)
    else:
        return trima_numba(close, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def trima_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    Add TRIMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"TRIMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with TRIMA column.
    """
    close = df[close_col].to_numpy()
    result = trima_ind(close, length, offset, fillna, use_talib)
    out_name = output_col or f"TRIMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])