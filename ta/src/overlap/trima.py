# -*- coding: utf-8 -*-
"""Triangular Moving Average (TRIMA) for financial time series.

TRIMA is a double-smoothed SMA whose weighting forms a triangle:
- odd ``length``:  SMA(SMA(x, h), h) with h = (length + 1) // 2
- even ``length``: SMA(SMA(x, h), h + 1) with h = length // 2

Both variants match TA-Lib's TRIMA kernel exactly.
"""
import numpy as np
import polars as pl

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available
from ..overlap.sma import _sma_numba_opt


# ----------------------------------------------------------------------
# TRIMA using Numba (double SMA, seeded on the valid part)
# ----------------------------------------------------------------------
def trima_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Triangular Moving Average using the Numba SMA core.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        TRIMA period (must be >= 1).
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        TRIMA values, same length as `close`. The first `length - 1`
        elements are NaN (warmup).

    Raises
    ------
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - The nested SMA is seeded on the valid (non-warmup) part of the
      first SMA; otherwise the NaN warmup head would poison the result.
    - Infinites in `close` are replaced with NaN before calculation.
    - With `nan_policy='ignore'` a NaN poisons the output from its
      position onward (IEEE 754 propagation).
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError('TRIMA length must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')

    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    n = len(close)
    trima = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or n < length:
        return _apply_offset_fillna(trima, offset, fillna)

    if length % 2 == 1:
        # Odd: double SMA with half = (length + 1) // 2.
        h = (length + 1) // 2
        sma1 = _sma_numba_opt(close, h)
        sma2_tail = _sma_numba_opt(
            np.ascontiguousarray(sma1[h - 1:]), h
        )
        trima[length - 1:] = sma2_tail[h - 1:]
    else:
        # Even: SMA(SMA(x, length/2), length/2 + 1).
        h = length // 2
        sma1 = _sma_numba_opt(close, h)
        sma2_tail = _sma_numba_opt(
            np.ascontiguousarray(sma1[h - 1:]), h + 1
        )
        trima[length - 1:] = sma2_tail[h:]

    return _apply_offset_fillna(trima, offset, fillna)


# ----------------------------------------------------------------------
# TRIMA using TA‑Lib (if available)
# ----------------------------------------------------------------------
def trima_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Triangular Moving Average via TA-Lib.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        TRIMA period (must be >= 1).
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        TRIMA values, same length as `close`.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    if length < 1:
        raise ValueError('TRIMA length must be >= 1')

    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')

    trima = talib.TRIMA(close, timeperiod=length)
    return _apply_offset_fillna(trima, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def trima_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Universal TRIMA with backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 10
        TRIMA period.
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    use_talib : bool, default True
        If True and TA‑Lib is available, use it; else use Numba.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        TRIMA values.

    Notes
    -----
    - With `nan_policy='ignore'` a NaN poisons the output from its
      position onward (IEEE 754 propagation).
    - This function is IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return trima_talib(close, length, offset, fillna, nan_policy)
    else:
        return trima_numba(close, length, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def trima_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None
) -> pl.DataFrame:
    """Add TRIMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str, default 'close'
        Column with close prices.
    length : int, default 10
        TRIMA period.
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    use_talib : bool, default True
        If True and TA‑Lib is available, use it; else use Numba.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.
    output_col : str, optional
        Output column name (default f"TRIMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with TRIMA column.

    """
    close = df[close_col].to_numpy()
    result = trima_ind(
        close, length, offset, fillna, use_talib, nan_policy
    )
    out_name = output_col or f'TRIMA_{length}'
    return df.with_columns([pl.Series(out_name, result)])