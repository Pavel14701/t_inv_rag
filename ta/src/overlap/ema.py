# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Optimized EMA using Numba (nopython mode, fastmath)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def _ema_numba_opt(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Exponential Moving Average (optimized Numba version).

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices.
    window : int
        EMA period.

    Returns
    -------
    np.ndarray
        EMA array with same length as input; first (window-1) values are NaN.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    alpha = 2.0 / (window + 1)
    s = 0.0
    for i in range(window):
        s += arr[i]
    out[window - 1] = s / window
    for i in range(window, n):
        out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return out


# ----------------------------------------------------------------------
# EMA using TA-Lib (when available) and Numba (optimized fallback)
# ----------------------------------------------------------------------
def ema_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """Exponential Moving Average using Numba (optimized)."""
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    ema = _ema_numba_opt(close, length)
    return _apply_offset_fillna(ema, offset, fillna)


def ema_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """EMA via TA-Lib."""
    if not talib_available:
        raise ImportError("TA-Lib not available")
    close = close.astype(np.float64)
    ema = talib.EMA(close, timeperiod=length)
    return _apply_offset_fillna(ema, offset, fillna)


def ema_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """Universal EMA with automatic implementation selection."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = close.astype(np.float64)
    if use_talib and talib_available:
        return ema_talib(close, length, offset, fillna)
    else:
        return ema_numba(close, length, offset, fillna)


def ema_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """EMA for Polars DataFrame."""
    close = df[close_col].to_numpy()
    result = ema_ind(
        close, 
        length=length, 
        offset=offset, 
        fillna=fillna, 
        use_talib=use_talib
    )
    out_name = output_col or f"EMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])