# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..utils import _apply_offset_fillna
from . import ema_ind


# ----------------------------------------------------------------------
# TEMA using Numba (triple EMA)
# ----------------------------------------------------------------------
def tema_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    TEMA using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    ema1 = ema_ind(close, length, offset=0, fillna=None, use_talib=False)
    ema2 = ema_ind(ema1, length, offset=0, fillna=None, use_talib=False)
    ema3 = ema_ind(ema2, length, offset=0, fillna=None, use_talib=False)
    tema = 3.0 * (ema1 - ema2) + ema3
    return _apply_offset_fillna(tema, offset, fillna)


# ----------------------------------------------------------------------
# TEMA using TA‑Lib (if available)
# ----------------------------------------------------------------------
def tema_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    TEMA using TA‑Lib (C implementation).
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    tema = talib.TEMA(close, timeperiod=length)
    return _apply_offset_fillna(tema, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def tema_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal TEMA with backend selection.
    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        EMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        If True and TA‑Lib is available, use it; else use Numba.
    Returns
    -------
    np.ndarray
        TEMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return tema_talib(close, length, offset, fillna)
    else:
        return tema_numba(close, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def tema_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    Add TEMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"TEMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with TEMA column.
    """
    close = df[close_col].to_numpy()
    result = tema_ind(close, length, offset, fillna, use_talib)
    out_name = output_col or f"TEMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])