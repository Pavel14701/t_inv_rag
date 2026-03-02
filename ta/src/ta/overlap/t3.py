# -*- coding: utf-8 -*-
"""
T3 Moving Average – Numba‑accelerated with TA‑Lib fallback.
"""

from typing import Optional

import numpy as np
import polars as pl

from .. import talib, talib_available
from ..overlap import ema_ind
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core T3 calculation using Numba (six‑fold EMA)
# ----------------------------------------------------------------------
def t3_numba(
    close: np.ndarray,
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    T3 moving average using Numba (raw numpy version).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        EMA period.
    a : float
        Volume factor (0 < a < 1).
    offset, fillna : as usual.

    Returns
    -------
    np.ndarray
        T3 values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Coefficients
    a2 = a * a
    a3 = a2 * a
    c1 = -a2 * a
    c2 = 3.0 * a2 + 3.0 * a3
    c3 = -6.0 * a2 - 3.0 * a - 3.0 * a3
    c4 = a3 + 3.0 * a2 + 3.0 * a + 1.0
    # Six successive EMAs
    e1 = ema_ind(close, length, use_talib=False)
    e2 = ema_ind(e1, length, use_talib=False)
    e3 = ema_ind(e2, length, use_talib=False)
    e4 = ema_ind(e3, length, use_talib=False)
    e5 = ema_ind(e4, length, use_talib=False)
    e6 = ema_ind(e5, length, use_talib=False)
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return _apply_offset_fillna(t3, offset, fillna)


# ----------------------------------------------------------------------
# TA‑Lib wrapper
# ----------------------------------------------------------------------
def t3_talib(
    close: np.ndarray,
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    T3 using TA‑Lib (C implementation).
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    t3 = talib.T3(close, timeperiod=length, vfactor=a)
    return _apply_offset_fillna(t3, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def t3_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal T3 moving average.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        EMA period.
    a : float
        Volume factor (0 < a < 1).
    offset, fillna : as usual.
    use_talib : bool
        Use TA‑Lib if available.

    Returns
    -------
    np.ndarray
        T3 values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return t3_talib(close, length, a, offset, fillna)
    else:
        return t3_numba(close, length, a, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def t3_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Add T3 column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, a, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"T3_{length}_{a}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with T3 column.
    """
    close = df[close_col].to_numpy()
    result = t3_ind(close, length, a, offset, fillna, use_talib)
    out_name = output_col or f"T3_{length}_{a}"
    return df.with_columns([pl.Series(out_name, result)])