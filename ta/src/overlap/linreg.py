# -*- coding: utf-8 -*-
from typing import Literal, Optional

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core Numba implementation (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _linreg_numba_core(
    close: np.ndarray,
    length: int,
    mode: str,
    degrees: bool
) -> np.ndarray:
    """
    Linear regression core for all modes.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.
    mode : str
        One of: 'line', 'tsf', 'slope', 'intercept', 'angle', 'r'.
    degrees : bool
        If mode='angle', return degrees instead of radians.

    Returns
    -------
    np.ndarray
        Result series; first (length-1) values are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Pre‑computed x values (1..length)
    x = np.arange(1, length + 1, dtype=np.float64)
    x_sum = x.sum()
    x2_sum = (x * x).sum()
    divisor = length * x2_sum - x_sum * x_sum
    inv_divisor = 1.0 / divisor if divisor != 0.0 else 0.0
    for i in range(length - 1, n):
        y = close[i - length + 1:i + 1]
        y_sum = y.sum()
        xy_sum = np.dot(x, y)
        if mode == 'r':
            y2_sum = (y * y).sum()
        # slope
        slope = (length * xy_sum - x_sum * y_sum) * inv_divisor
        if mode == 'slope':
            out[i] = slope
            continue
        # intercept
        intercept = (y_sum - slope * x_sum) / length
        if mode == 'intercept':
            out[i] = intercept
            continue
        # angle
        if mode == 'angle':
            angle = np.arctan(slope)
            if degrees:
                angle *= 180.0 / np.pi
            out[i] = angle
            continue
        # correlation
        if mode == 'r':
            y2_sum = (y * y).sum()
            denom = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
            if denom != 0.0:
                out[i] = (length * xy_sum - x_sum * y_sum) / denom
            else:
                out[i] = 0.0
            continue
        # line or tsf
        if mode == 'tsf':
            x_last = length + 1.0  # next point
        else:  # 'line'
            x_last = length        # last point
        out[i] = slope * x_last + intercept

    return out


# ----------------------------------------------------------------------
# TA‑Lib wrapper (where available)
# ----------------------------------------------------------------------
def linreg_talib(
    close: np.ndarray,
    length: int,
    mode: str,
    degrees: bool = False
) -> np.ndarray:
    """
    Linear regression using TA‑Lib (
    only for modes where TA‑Lib has dedicated functions).
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if mode == 'line':
        result = talib.LINEARREG(close, timeperiod=length)
    elif mode == 'tsf':
        result = talib.TSF(close, timeperiod=length)
    elif mode == 'slope':
        result = talib.LINEARREG_SLOPE(close, timeperiod=length)
    elif mode == 'intercept':
        result = talib.LINEARREG_INTERCEPT(close, timeperiod=length)
    elif mode == 'angle':
        result = talib.LINEARREG_ANGLE(close, timeperiod=length)
        if degrees:
            result = result * 180.0 / np.pi
    else:
        raise ValueError(f"Mode '{mode}' not supported by TA‑Lib")
    return result


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def linreg_numba(
    close: np.ndarray,
    length: int = 14,
    mode: Literal['line', 'tsf', 'slope', 'intercept', 'angle', 'r'] = 'line',
    degrees: bool = False,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Linear regression using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        Window length.
    mode : str
        One of: 'line' (value at last point), 'tsf' (forecast next point),
        'slope', 'intercept', 'angle', 'r' (correlation).
    degrees : bool
        If mode='angle', return degrees instead of radians.
    offset, fillna : as usual.

    Returns
    -------
    np.ndarray
        Result series.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    result = _linreg_numba_core(close, length, mode, degrees)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def linreg_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    mode: Literal['line', 'tsf', 'slope', 'intercept', 'angle', 'r'] = 'line',
    degrees: bool = False,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal Linear Regression.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Window length.
    mode : str
        One of: 'line', 'tsf', 'slope', 'intercept', 'angle', 'r'.
    degrees : bool
        If mode='angle', return degrees instead of radians.
    offset : int
        Shift result.
    fillna : float, optional
        Fill NaN with this value.
    use_talib : bool
        Use TA‑Lib if available and mode is supported.
    Returns
    -------
    np.ndarray
        Result series.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    if use_talib and talib_available and mode != 'r':
        result = linreg_talib(close, length, mode, degrees)
        return _apply_offset_fillna(result, offset, fillna)
    else:
        return linreg_numba(close, length, mode, degrees, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def linreg_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    mode: Literal['line', 'tsf', 'slope', 'intercept', 'angle', 'r'] = 'line',
    degrees: bool = False,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Add linear regression column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length : int
        Window length.
    mode : str
        As above.
    degrees : bool
        As above.
    offset, fillna : as usual.
    use_talib : bool
        Use TA‑Lib if available.
    output_col : str, optional
        Output column name. Default: f"LINREG_{mode}_{length}".

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    result = linreg_ind(close, length, mode, degrees, offset, fillna, use_talib)
    out_name = output_col or f"LINREG_{mode}_{length}"
    return df.with_columns([pl.Series(out_name, result)])