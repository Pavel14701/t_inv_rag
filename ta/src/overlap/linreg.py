# -*- coding: utf-8 -*-
"""Linear Regression indicator (LINREG) implementation.

This module provides:
- Numba-accelerated core (`_linreg_numba_core`)
- Numba wrapper (`linreg_numba`), TA-Lib backend (`linreg_talib`)
- Universal wrapper (`linreg_ind`)
- Polars integration (`linreg_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

from typing import Literal, Optional

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available


# ----------------------------------------------------------------------
# Core Numba implementation (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _linreg_numba_core(
    close: np.ndarray, length: int, mode: str, degrees: bool
) -> np.ndarray:
    """Linear regression core for all modes.

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
    # Pre-computed x values (1..length)
    x = np.arange(1, length + 1, dtype=np.float64)
    x_sum = x.sum()
    x2_sum = (x * x).sum()
    divisor = length * x2_sum - x_sum * x_sum
    inv_divisor = 1.0 / divisor if divisor != 0.0 else 0.0  # noqa: RUF069 - exact IEEE zero/sign check
    for i in range(length - 1, n):
        y = close[i - length + 1 : i + 1]
        y_sum = y.sum()
        xy_sum = np.dot(x, y)
        # slope
        slope = (length * xy_sum - x_sum * y_sum) * inv_divisor
        if mode == "slope":
            out[i] = slope
            continue
        # intercept
        intercept = (y_sum - slope * x_sum) / length
        if mode == "intercept":
            out[i] = intercept
            continue
        # angle
        if mode == "angle":
            angle = np.arctan(slope)
            if degrees:
                angle *= 180.0 / np.pi
            out[i] = angle
            continue
        # correlation
        if mode == "r":
            y2_sum = (y * y).sum()
            denom = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
            if denom != 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
                out[i] = (length * xy_sum - x_sum * y_sum) / denom
            else:
                out[i] = 0.0
            continue
        # line or tsf
        if mode == "tsf":
            x_last = length + 1.0  # next point
        else:  # 'line'
            x_last = length  # last point
        out[i] = slope * x_last + intercept

    return out


# ----------------------------------------------------------------------
# TA-Lib wrapper (where available)
# ----------------------------------------------------------------------
def linreg_talib(
    close: np.ndarray, length: int, mode: str, degrees: bool = False
) -> np.ndarray:
    """Linear regression using TA-Lib (
    only for modes where TA-Lib has dedicated functions).
    """
    if not talib_available:
        raise ImportError("TA-Lib not available")
    if length < 1:
        raise ValueError("LINREG length must be >= 1")
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Replace infinities with NaN (IEEE 754 compliance)
    close = close.copy()
    replace_inf_with_nan(close)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if mode == "line":
        result = talib.LINEARREG(close, timeperiod=length)
    elif mode == "tsf":
        result = talib.TSF(close, timeperiod=length)
    elif mode == "slope":
        result = talib.LINEARREG_SLOPE(close, timeperiod=length)
    elif mode == "intercept":
        result = talib.LINEARREG_INTERCEPT(close, timeperiod=length)
    elif mode == "angle":
        result = talib.LINEARREG_ANGLE(close, timeperiod=length)
        # TA-Lib LINEARREG_ANGLE already returns degrees. Convert to
        # radians only when degrees=False, to match the Numba backend.
        if not degrees:
            result = result * np.pi / 180.0
    else:
        raise ValueError(f"Mode '{mode}' not supported by TA-Lib")
    return result


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def linreg_numba(
    close: np.ndarray,
    length: int = 14,
    mode: Literal["line", "tsf", "slope", "intercept", "angle", "r"] = "line",
    degrees: bool = False,
    offset: int = 0,
    fillna: Optional[float] = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Linear regression using Numba.

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
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    np.ndarray
        Result series.

    Raises
    ------
    ValueError
        If `length < 1`, `mode` is unknown, the input contains NaN with
        `nan_policy='raise'`, or `nan_policy` is unknown.

    Notes
    -----
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError("LINREG length must be >= 1")
    if mode not in ("line", "tsf", "slope", "intercept", "angle", "r"):
        raise ValueError(f"Unsupported mode: {mode}")
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Replace infinities with NaN (IEEE 754 compliance)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")
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
    mode: Literal["line", "tsf", "slope", "intercept", "angle", "r"] = "line",
    degrees: bool = False,
    offset: int = 0,
    fillna: Optional[float] = None,
    nan_policy: str = "raise",
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Linear Regression.

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
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`.
    use_talib : bool
        Use TA-Lib if available and mode is supported.

    Returns
    -------
    np.ndarray
        Result series.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    if use_talib and talib_available and mode != "r":
        result = linreg_talib(close, length, mode, degrees)
        return _apply_offset_fillna(result, offset, fillna)
    else:
        return linreg_numba(
            close, length, mode, degrees, offset, fillna, nan_policy
        )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def linreg_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    mode: Literal["line", "tsf", "slope", "intercept", "angle", "r"] = "line",
    degrees: bool = False,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: Optional[str] = None,
) -> pl.DataFrame:
    """Add linear regression column to Polars DataFrame.

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
        Use TA-Lib if available.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str, optional
        Output column name. Default: f"LINREG_{mode}_{length}".

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = linreg_ind(
        close,
        length=length,
        mode=mode,
        degrees=degrees,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"LINREG_{mode}_{length}"
    return df.with_columns([pl.Series(out_name, result)])
