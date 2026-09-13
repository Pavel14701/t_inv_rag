# -*- coding: utf-8 -*-
"""Fibonacci Weighted Moving Average (FWMA) implementation.

FWMA assigns weights based on the Fibonacci sequence: the oldest value gets
weight 1, the next gets 1, then 2, 3, 5, etc. The weights can be applied in
ascending order (recent values get higher weight) or descending.

This module provides:
- Cached Fibonacci weight generation (`_get_fib_weights`)
- Numba-accelerated core (`_fwma_numba_cached`)
- Numba wrapper with offset/fillna (`fwma_numba`)
- Universal wrapper (`fwma_ind`)
- Polars integration (`fwma_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

from functools import lru_cache

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)


# ----------------------------------------------------------------------
# Cached weights
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _get_fib_weights(length: int, asc: bool) -> np.ndarray:
    """Generate normalized Fibonacci weights for a given window length.

    Parameters
    ----------
    length : int
        Number of weights (window size). Must be >= 1.
    asc : bool
        If True, weights increase towards the end (recent values have higher
        weight). If False, weights decrease towards the end.

    Returns
    -------
    np.ndarray
        1D float64 array of length `length` summing to 1.

    Notes
    -----
    - The weights are generated using the
        Fibonacci recurrence: 1, 1, 2, 3, 5, ...
    - The result is cached via `lru_cache` to avoid recomputation.
    - For `length=1`, the weight is [1.0].

    """
    w = np.zeros(length, dtype=np.float64)
    w[0] = 1.0
    if length > 1:
        w[1] = 1.0
        for i in range(2, length):
            w[i] = w[i - 1] + w[i - 2]
    if not asc:
        w = w[::-1]
    w /= w.sum()
    # Protect the lru_cache from accidental in-place modification
    w.flags.writeable = False
    return w


# ----------------------------------------------------------------------
# Numba core
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _fwma_numba_cached(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """FWMA core loop with precomputed weights (Numba).

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices (assumed to have no NaNs or infinities).
    weights : np.ndarray
        Normalized weights of length `window` (summing to 1).

    Returns
    -------
    np.ndarray
        FWMA array with same length as `arr`; first `len(weights)-1` values
        are NaN (insufficient data).

    Notes
    -----
    - This function is called by `fwma_numba` after NaN/Inf handling.
    - The convolution is applied with weights aligned so that the most recent
        value in the window gets `weights[-1]`.

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
# Numba wrapper with offset/fillna
# ----------------------------------------------------------------------
def fwma_numba(
    close: np.ndarray,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Fibonacci Weighted Moving Average
    using Numba (fallback/primary backend).

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of closing prices.
    length : int, default 10
        FWMA window length. Must be >= 1.
    asc : bool, default True
        If True, recent values get higher weight; if False, older values get
        higher weight.
    offset : int, default 0
        Shift the result. Positive = forward (shifted to the future),
        negative = backward (shifted to the past).
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        FWMA values, same length as `close`.

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
        raise ValueError("FWMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64, copy=False)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    weights = _get_fib_weights(length, asc)
    fwma = _fwma_numba_cached(close, weights)
    return _apply_offset_fillna(fwma, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def fwma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal FWMA (always uses Numba, no TA-Lib equivalent).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Input price data.
    length : int, default 10
        FWMA window length.
    asc : bool, default True
        Weight direction (True = recent higher weight, False = older higher).
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        FWMA values, same length as `close`.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return fwma_numba(close, length, asc, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def fwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add FWMA column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 10
        FWMA window length.
    asc : bool, default True
        Weight direction (True = recent higher weight, False = older higher).
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str or None, default None
        Name of the output column. If None, defaults to f'FWMA_{length}'.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with an additional column containing FWMA values.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = fwma_ind(
        close,
        length=length,
        asc=asc,
        offset=offset,
        fillna=fillna,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"FWMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])
