# -*- coding: utf-8 -*-
"""
Kaufman's Adaptive Moving Average (KAMA) with dual backend (Numba/TA‑Lib).
"""

from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core KAMA calculation in Numba (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _kama_numba_core(
    close: np.ndarray,
    length: int,
    fast: int,
    slow: int,
    drift: int
) -> np.ndarray:
    """
    KAMA core loop (Numba implementation).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Period for efficiency ratio.
    fast, slow : int
        Fast and slow EMA periods.
    drift : int
        Shift for price difference.

    Returns
    -------
    np.ndarray
        KAMA values; first `length-1` positions are NaN.
    """
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return kama

    # Constants
    fr = 2.0 / (fast + 1)
    sr = 2.0 / (slow + 1)

    # Pre‑compute absolute differences over drift (for peer_diff_sum)
    abs_drift = np.empty(n, dtype=np.float64)
    for i in range(1, n):
        abs_drift[i] = abs(close[i] - close[i - drift])
    abs_drift[0] = 0.0

    # Cumulative sum of abs_drift
    cum = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        cum[i] = cum[i - 1] + abs_drift[i - 1]

    # Initial KAMA value (SMA of first `length` values)
    s = 0.0
    for i in range(length):
        s += close[i]
    kama[length - 1] = s / length

    # Main recurrence
    for i in range(length, n):
        abs_diff = abs(close[i] - close[i - length])
        peer_sum = cum[i + 1] - cum[i + 1 - length]
        er = 0.0 if peer_sum == 0.0 else abs_diff / peer_sum
        sc = (er * (fr - sr) + sr) ** 2
        kama[i] = sc * close[i] + (1.0 - sc) * kama[i - 1]

    return kama


# ----------------------------------------------------------------------
# KAMA via TA-Lib (if available)
# ----------------------------------------------------------------------
def kama_talib(
    close: np.ndarray,
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    KAMA using TA-Lib (C implementation).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Period for efficiency ratio.
    fast : int
        Fast EMA period.
    slow : int
        Slow EMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        KAMA values.
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    kama = talib.KAMA(close, timeperiod=length)
    return _apply_offset_fillna(kama, offset, fillna)


# ----------------------------------------------------------------------
# Public functions (Numba + Polars)
# ----------------------------------------------------------------------
def kama_numba(
    close: np.ndarray,
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
    drift: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    KAMA using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    kama = _kama_numba_core(close, length, fast, slow, drift)
    return _apply_offset_fillna(kama, offset, fillna)


def kama_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
    drift: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal KAMA with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Period for efficiency ratio.
    fast : int
        Fast EMA period.
    slow : int
        Slow EMA period.
    drift : int
        Shift for price difference (only used in Numba version).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        If True and TA-Lib is available, use it; else use Numba.

    Returns
    -------
    np.ndarray
        KAMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return kama_talib(close, length, fast, slow, offset, fillna)
    else:
        return kama_numba(close, length, fast, slow, drift, offset, fillna)


def kama_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
    drift: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    KAMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Column name for close prices.
    length : int
        Period for efficiency ratio.
    fast : int
        Fast EMA period.
    slow : int
        Slow EMA period.
    drift : int
        Shift for price difference (only used in Numba version).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA-Lib if available.
    output_col : str, optional
        Output column name (default f"KAMA_{length}_{fast}_{slow}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with KAMA series.
    """
    close = df[close_col].to_numpy()
    result = kama_ind(close, length, fast, slow, drift, offset, fillna, use_talib)
    out_name = output_col or f"KAMA_{length}_{fast}_{slow}"
    return df.with_columns([pl.Series(out_name, result)])