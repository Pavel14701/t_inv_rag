# -*- coding: utf-8 -*-
"""Smoothed Moving Average (SMMA) implementation.

This module provides:
- Numba-accelerated core (`_smma_numba_core`)
- Numba wrapper with NaN handling, offset/fillna (`smma_numba`)
- Universal wrapper (`smma_ind`)
- Polars integration (`smma_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)


# ----------------------------------------------------------------------
# Core Numba implementation of SMMA
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True, fastmath=False)
def _smma_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """Smoothed Moving Average (SMMA) core calculation.

    First value (at index length-1) is SMA of first `length` elements.
    Then: SMMA[i] = ((length-1) * SMMA[i-1] + close[i]) / length

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of prices (assumed to have no NaNs or infinities).
    length : int
        SMMA period.

    Returns
    -------
    np.ndarray
        SMMA array; first (length-1) values are NaN.

    Notes
    -----
    - This function assumes `close` has no NaNs or infinities.
    - NaN/Inf handling is done in the caller.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial SMA
    s = 0.0
    for i in range(length):
        s += close[i]
    out[length - 1] = s / length
    # Recurrence
    for i in range(length, n):
        out[i] = ((length - 1) * out[i - 1] + close[i]) / length
    return out


# ----------------------------------------------------------------------
# SMMA using Numba (with NaN handling, offset and fillna)
# ----------------------------------------------------------------------
def smma_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Smoothed Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int, default 10
        SMMA period (must be >= 1).
    offset : int, default 0
        Shift result. Positive = forward, negative = backward.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        SMMA values, same length as `close`.

    Raises
    ------
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - The first `length-1` elements are NaN because the window is not full.
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError("SMMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Replace infinities with NaN (IEEE 754 compliance)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _smma_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal SMMA function (always uses Numba)
# ----------------------------------------------------------------------
def smma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal Smoothed Moving Average (Numba only).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 10
        SMMA period.
    offset : int, default 0
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`.

    Returns
    -------
    np.ndarray
        SMMA values.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return smma_numba(close, length, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def smma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """SMMA for Polars DataFrame (Numba only).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column with close prices.
    length : int, default 10
        SMMA period.
    offset : int, default 0
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str, optional
        Output column name (default f"SMMA_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = smma_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"SMMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])
