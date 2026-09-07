# -*- coding: utf-8 -*-
"""Sine Weighted Moving Average (SINWMA) implementation.

Weights follow the canonical definition:
w_i = sin(i * pi / (length + 1)), i = 1..length, normalized to sum 1.
The weights are symmetric by design (same as Everget's Pine script).

This module provides:
- Numba-accelerated core (`_sinwma_numba_core`)
- Numba wrapper with NaN handling, offset/fillna (`sinwma_numba`)
- Universal wrapper (`sinwma_ind`)
- Polars integration (`sinwma_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""
from functools import lru_cache
from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)


# ----------------------------------------------------------------------
# Cached sine weights
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _sine_weights(length: int) -> np.ndarray:
    """Generate normalized sine weights for SINWMA.

    Formula: w_i = sin(i * pi / (length+1)) for i = 1..length,
    then normalized to sum to 1.

    Parameters
    ----------
    length : int
        Window size (must be >= 1).

    Returns
    -------
    np.ndarray
        Normalised weights summing to 1 (read-only).

    Raises
    ------
    ValueError
        If `length < 1`.

    """
    if length < 1:
        raise ValueError(f'SINWMA length must be >= 1, got {length}')
    i = np.arange(1, length + 1, dtype=np.float64)
    w = np.sin(i * np.pi / (length + 1))
    w /= w.sum()
    # Protect the lru_cache from accidental in-place modification
    w.flags.writeable = False
    return w


# ----------------------------------------------------------------------
# Core SINWMA calculation in Numba (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _sinwma_numba_core(close: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """SINWMA core loop.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    weights : np.ndarray
        Normalized sine weights (length = window size).

    Returns
    -------
    np.ndarray
        SINWMA values; first (len(weights)-1) positions are NaN.

    """
    n = len(close)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += close[i - j] * weights[length - 1 - j]
        out[i] = acc
    return out


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def sinwma_numba(
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: Optional[float] = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Sine Weighted Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int, default 14
        Window length (must be >= 1).
    offset : int, default 0
        Shift result. Positive = forward, negative = backward.
    fillna : float, optional
        Value to fill NaNs and shifted-in positions.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        SINWMA values; first (length - 1) positions are NaN (or `fillna`).

    Raises
    ------
    ValueError
        If `length < 1`, `nan_policy` is unknown, the input contains NaN
        with `nan_policy='raise'`, or the series is too short.

    Notes
    -----
    - Infinite values in `close` are replaced with NaN.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError(f'SINWMA length must be >= 1, got {length}')
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

    weights = _sine_weights(length)
    result = _sinwma_numba_core(close, weights)

    return _apply_offset_fillna(result, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def sinwma_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    offset: int = 0,
    fillna: Optional[float] = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Universal SINWMA (always uses Numba).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 14
        Window length (must be >= 1).
    offset : int, default 0
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`.

    Returns
    -------
    np.ndarray
        SINWMA values.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return sinwma_numba(close, length, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def sinwma_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 14,
    offset: int = 0,
    fillna: Optional[float] = None,
    nan_policy: str = 'raise',
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """Add SINWMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str, optional
        Output column name (default f"SINWMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with SINWMA column.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = sinwma_ind(
        close, length, offset, fillna, nan_policy
    )
    out_name = output_col or f'SINWMA_{length}'
    return df.with_columns([pl.Series(out_name, result)])