# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import polars as pl
from numba import float64, njit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna, _handle_nan_policy


# ----------------------------------------------------------------------
# Cached weights for WMA (linear weights)
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _get_wma_weights(length: int, asc: bool) -> np.ndarray:
    """
    Generate normalized linear weights for WMA.
    If asc=True, weights increase from 1 to length (most recent heaviest).
    If asc=False, weights decrease (most recent lightest).
    Weights are normalized so that sum = 1.
    """
    w = np.arange(1, length + 1, dtype=np.float64)
    if not asc:
        w = w[::-1]
    w /= w.sum()
    return w


# ----------------------------------------------------------------------
# WMA core loop (Numba) with typed signature
# ----------------------------------------------------------------------
@njit((float64[:], float64[:]), fastmath=True, cache=True)
def _wma_numba_core(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted Moving Average core loop.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array (assumed to have no NaNs).
    weights : np.ndarray
        Normalized weights (length = window size).

    Returns
    -------
    np.ndarray
        WMA values; first (len(weights)-1) positions are NaN.
    """
    n = len(arr)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        acc = 0.0
        # weighted sum over the window
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
    """
    Weighted Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        WMA period (>= 1).
    asc : bool
        If True, recent values have higher weight (default).
        If False, older values have higher weight.
    offset, fillna, nan_policy, trim : as usual.

    Returns
    -------
    np.ndarray
        WMA values.
    """
    # ---- Input validation ----
    if length < 1:
        raise ValueError("WMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")
    # Apply NaN policy
    close = _handle_nan_policy(close, nan_policy, "close")
    # Ensure C-contiguous
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Check length
    if len(close) < length:
        raise ValueError(
            f"Input series too short: need at least \
                {length} elements, got {len(close)}."
            )
    # Get weights and compute WMA
    weights = _get_wma_weights(length, asc)
    wma = _wma_numba_core(close, weights)
    # Trim if requested
    if trim:
        valid_start = length - 1
        if valid_start < len(wma):
            wma = wma[valid_start:]
        else:
            wma = np.array([])
    # Apply offset and fillna
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
    """
    Weighted Moving Average via TA-Lib (asc=True only).

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        WMA period (>= 1).
    offset, fillna, nan_policy, trim : as usual.
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    if length < 1:
        raise ValueError("WMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")
    # TA‑Lib doesn't handle NaNs, so pre-process
    close = _handle_nan_policy(close, nan_policy, "close")
    if len(close) < length:
        raise ValueError(
            f"Input series too short: need at least \
                {length} elements, got {len(close)}."
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
    """
    Universal Weighted Moving Average with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        WMA period.
    asc : bool
        If True, recent values have higher weight (TA-Lib compatible).
        If False, older values have higher weight (Numba only).
    offset, fillna, use_talib, nan_policy, trim : as usual.

    Returns
    -------
    np.ndarray
        WMA values.
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
    close_col: str = "close",
    length: int = 10,
    asc: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    WMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length, asc, offset, fillna, use_talib, nan_policy : as in wma_ind.
    output_col : str, optional
        Output column name (default f"WMA_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added column (same length).
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
        trim=False,  # Polars всегда возвращает полную длину
    )
    out_name = output_col or f"WMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])