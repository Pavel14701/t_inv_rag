# -*- coding: utf-8 -*-
"""Exponential Moving Average (EMA) implementation.

Provides:
- Numba-accelerated core (`_ema_numba_opt`)
- Numba wrapper with NaN handling, offset, fillna, trim (`ema_numba`)
- TA-Lib backend (`ema_talib`)
- Universal wrapper (`ema_ind`)
- Polars integration (`ema_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""
import numpy as np
import polars as pl
from numba import jit

from ..external import talib, talib_available
from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan
)


# ----------------------------------------------------------------------
# Optimized EMA using Numba (nopython mode, fastmath=False)
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True, parallel=False, fastmath=False)
def _ema_numba_opt(arr: np.ndarray, window: int) -> np.ndarray:
    """Exponential Moving Average (optimized Numba version).

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices (assumed to have no NaNs).
    window : int
        EMA period.

    Returns
    -------
    np.ndarray
        EMA array with same length as input; first (window-1) values are NaN.

    Notes
    -----
    - This function assumes `arr` has no NaNs or infinities.
    - NaN/Inf handling is done in the caller.

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
# EMA using Numba (with NaN handling, trim, etc.)
# ----------------------------------------------------------------------
def ema_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Exponential Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int, default 10
        EMA period.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values: 'raise', 'ignore', 'ffill', 'bfill', 'both'.
    trim : bool, default False
        If True, remove the first (length-1) elements (which are NaN).

    Returns
    -------
    np.ndarray
        EMA array.

    Raises
    ------
    ValueError
        If `length < 1` or `nan_policy == 'raise'` and NaNs are present.

    Notes
    -----
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError('EMA length must be >= 1')
    close = np.asarray(close, dtype=np.float64)
    # Replace infinities with NaN (IEEE 754 compliance)
    close = close.copy()
    replace_inf_with_nan(close)
    # Apply NaN policy
    close = _handle_nan_policy(close, nan_policy, 'close')
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if len(close) < length:
        raise ValueError(
            f'Input series too short: need at least {length} elements, '
            f'got {len(close)}.'
        )
    ema = _ema_numba_opt(close, length)
    if trim:
        valid_start = length - 1
        if valid_start < len(ema):
            ema = ema[valid_start:]
        else:
            ema = np.array([])
    return _apply_offset_fillna(ema, offset, fillna)


# ----------------------------------------------------------------------
# EMA using TA-Lib (with NaN handling – TA-Lib itself doesn't handle NaNs)
# ----------------------------------------------------------------------
def ema_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """EMA via TA-Lib, with pre-processing of NaNs and infinities.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int, default 10
        EMA period.
    offset, fillna, nan_policy, trim :
        Same as in `ema_numba`.

    Returns
    -------
    np.ndarray
        EMA array.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.
    ValueError
        If `length < 1` or `nan_policy == 'raise'` and NaNs are present.

    Notes
    -----
    - TA-Lib does not handle NaNs, so they are pre-processed.
    - Infinites are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    if length < 1:
        raise ValueError('EMA length must be >= 1')
    close = np.asarray(close, dtype=np.float64)
    # Replace infinities with NaN
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')
    if len(close) < length:
        raise ValueError(
            f'Input series too short: need at least {length} elements, '
            f'got {len(close)}.'
        )
    ema = talib.EMA(close, timeperiod=length)
    if trim:
        valid_start = length - 1
        if valid_start < len(ema):
            ema = ema[valid_start:]
        else:
            ema = np.array([])
    return _apply_offset_fillna(ema, offset, fillna)


# ----------------------------------------------------------------------
# Universal EMA function
# ----------------------------------------------------------------------
def ema_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Universal EMA with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Input price data.
    length : int, default 10
        EMA period.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    use_talib : bool, default True
        If True and TA-Lib is installed, use TA-Lib.
        Otherwise, fall back to the Numba implementation.
    nan_policy : str, default 'raise'
        How to handle NaN values.
    trim : bool, default False
        If True, remove the first (length-1) elements.

    Returns
    -------
    np.ndarray
        EMA array.

    Notes
    -----
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = close.astype(np.float64)

    if use_talib and talib_available:
        return ema_talib(
            close,
            length=length,
            offset=offset,
            fillna=fillna,
            nan_policy=nan_policy,
            trim=trim,
        )
    else:
        return ema_numba(
            close,
            length=length,
            offset=offset,
            fillna=fillna,
            nan_policy=nan_policy,
            trim=trim,
        )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def ema_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add EMA column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column with close prices.
    length : int, default 10
        EMA period.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    use_talib : bool, default True
        Use TA-Lib if available.
    nan_policy : str, default 'raise'
        How to handle NaN values.
    output_col : str or None, default None
        Name of the output column. If None, defaults to f'EMA_{length}'.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with an additional column containing EMA values.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = ema_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,
    )
    out_name = output_col or f'EMA_{length}'
    return df.with_columns([pl.Series(out_name, result)])
