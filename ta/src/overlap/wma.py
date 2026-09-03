# -*- coding: utf-8 -*-
"""Weighted Moving Average (WMA) implementation.

WMA assigns linearly increasing (or decreasing) weights to
prices in the window. The weight of the most recent
price is `length` (or 1 if `asc=False`).

This module provides:
- Numba-accelerated implementation (`wma_numba`)
- TA-Lib backend (`wma_talib`) – only for `asc=True`
- Universal wrapper (`wma_ind`)
- Polars integration (`wma_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

from functools import lru_cache

import numpy as np
import polars as pl
from numba import njit

from ..external import talib, talib_available
from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan
)


# ----------------------------------------------------------------------
# Cached weights for WMA (linear weights)
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _get_wma_weights(length: int, asc: bool) -> np.ndarray:
    """Generate normalized linear weights for WMA.

    Parameters
    ----------
    length : int
        Window size.
    asc : bool
        If True, weights increase from 1 to `length` (most recent heaviest).
        If False, weights decrease (most recent lightest).

    Returns
    -------
    np.ndarray
        Normalised weights summing to 1.

    Notes
    -----
    - Results are cached via `lru_cache` for performance.

    """
    w = np.arange(1, length + 1, dtype=np.float64)
    if not asc:
        w = w[::-1]
    w /= w.sum()
    # Protect the lru_cache from accidental in-place modification
    w.flags.writeable = False
    return w


# ----------------------------------------------------------------------
# WMA core loop (Numba) with typed signature
# ----------------------------------------------------------------------
@njit(fastmath=False, cache=True)
def _wma_numba_core(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Moving Average core loop.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array (assumed to have no NaNs or infinities).
    weights : np.ndarray
        Normalized weights (length = window size).

    Returns
    -------
    np.ndarray
        WMA values; first `len(weights)-1` positions are NaN.

    """
    n = len(arr)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += arr[i - j] * weights[length - 1 - j]
        out[i] = acc
    return out


# ----------------------------------------------------------------------
# WMA using Numba (with NaN handling, trim, etc.)
# ----------------------------------------------------------------------
def wma_numba(
    close: np.ndarray,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Weighted Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        WMA period (must be >= 1).
    asc : bool, default True
        If True, recent values have higher weight (default).
        If False, older values have higher weight.
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.
    nan_policy : str, default 'raise'
        How to handle NaN values: 'raise', 'ignore', 'ffill', 'bfill', 'both'.
    trim : bool, default False
        If True, remove the first `length-1` elements before
        applying `offset`.

    Returns
    -------
    np.ndarray
        WMA values, shifted and NaN-filled as requested.

    Raises
    ------
    ValueError
        If `length < 1`, invalid `nan_policy`, or series too short.

    Notes
    -----
    - Infinites in `close` are replaced with NaN.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError('WMA length must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)

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

    weights = _get_wma_weights(length, asc)
    wma = _wma_numba_core(close, weights)

    if trim:
        valid_start = length - 1
        if valid_start < len(wma):
            wma = wma[valid_start:]
        else:
            wma = np.array([])

    return _apply_offset_fillna(wma, offset, fillna)


# ----------------------------------------------------------------------
# WMA via TA-Lib (only asc=True) with NaN handling
# ----------------------------------------------------------------------
def wma_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Weighted Moving Average via TA-Lib (asc=True only).

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        WMA period.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs and shifted-in positions.
    nan_policy : str, default 'raise'
        How to handle NaN values: 'raise', 'ignore', 'ffill', 'bfill', 'both'.
    trim : bool, default False
        If True, remove the first `length-1` elements before
        applying `offset`.

    Returns
    -------
    np.ndarray
        WMA values.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.
    ValueError
        If `length < 1`, invalid `nan_policy`, or series too short.

    Notes
    -----
    - TA-Lib does not handle NaNs, so they are pre-processed.
    - Infinites are replaced with NaN.
    - Only `asc=True` is supported; `asc=False` falls back to Numba.

    """
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    if length < 1:
        raise ValueError('WMA length must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')
    if len(close) < length:
        raise ValueError(
            f'Input series too short: need at least {length} elements, '
            f'got {len(close)}.'
        )
    wma = talib.WMA(close, timeperiod=length)
    if trim:
        valid_start = length - 1
        if valid_start < len(wma):
            wma = wma[valid_start:]
        else:
            wma = np.array([])
    return _apply_offset_fillna(wma, offset, fillna)


# ----------------------------------------------------------------------
# Universal WMA function
# ----------------------------------------------------------------------
def wma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Universal Weighted Moving Average with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 10
        WMA period (must be >= 1).
    asc : bool, default True
        If True, recent values have higher weight (TA-Lib compatible).
        If False, older values have higher weight (Numba only).
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.
    use_talib : bool, default True
        If True and TA-Lib is installed, use TA-Lib (only when `asc=True`).
    nan_policy : str, default 'raise'
        How to handle NaN values: 'raise', 'ignore', 'ffill', 'bfill', 'both'.
    trim : bool, default False
        If True, remove the first `length-1` elements before
        applying `offset`.

    Returns
    -------
    np.ndarray
        WMA values, shifted and NaN-filled as requested.

    Raises
    ------
    ValueError
        If `length < 1`, invalid `nan_policy`, or series too short.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - TA-Lib is used only when `asc=True` and `use_talib=True`.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    if use_talib and talib_available and asc:
        return wma_talib(
            close,
            length=length,
            offset=offset,
            fillna=fillna,
            nan_policy=nan_policy,
            trim=trim,
        )
    else:
        return wma_numba(
            close,
            length=length,
            asc=asc,
            offset=offset,
            fillna=fillna,
            nan_policy=nan_policy,
            trim=trim,
        )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def wma_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add WMA column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 10
        WMA period.
    asc : bool, default True
        Weight direction (True = recent higher weight, False = older higher).
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs and shifted-in positions.
    use_talib : bool, default True
        If True and TA-Lib is available, use TA-Lib (only when `asc=True`).
    nan_policy : str, default 'raise'
        How to handle NaN values.
    output_col : str or None, default None
        Name of the output column. If None, defaults to f'WMA_{length}'.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with an additional column containing WMA values.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = wma_ind(
        close,
        length=length,
        asc=asc,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars always returns full length
    )
    out_name = output_col or f'WMA_{length}'
    return df.with_columns([pl.Series(out_name, result)])
