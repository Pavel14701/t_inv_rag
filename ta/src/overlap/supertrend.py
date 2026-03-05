# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib_available
from ..utils import _apply_offset_fillna
from ..volatility import atr_ind


# ----------------------------------------------------------------------
# Core Supertrend logic in Numba (single pass)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _supertrend_numba_core(
    close: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    initial_dir: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Supertrend loop.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    lb, ub : np.ndarray
        Lower and upper bands (pre‑computed).
    initial_dir : int
        Initial direction (1 for up, -1 for down). Usually set to 1.

    Returns
    -------
    trend, direction, long, short : np.ndarray
        All arrays of same length as close. First `length` values may be NaN.
    """
    n = len(close)
    trend = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.int64)
    long = np.full(n, np.nan, dtype=np.float64)
    short = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return trend, direction, long, short
    # Initial direction for index 0 (not used in final result)
    dir_val = initial_dir
    for i in range(1, n):
        # Determine direction
        if close[i] > ub[i - 1]:
            dir_val = 1
        elif close[i] < lb[i - 1]:
            dir_val = -1
        # else dir_val unchanged
        direction[i] = dir_val
        # Adjust bands based on direction
        if dir_val > 0:
            # Uptrend: lower band should not decrease
            lb[i] = max(lb[i], lb[i - 1])
            trend[i] = lb[i]
            long[i] = lb[i]
            short[i] = np.nan
        else:
            # Downtrend: upper band should not increase
            ub[i] = min(ub[i], ub[i - 1])
            trend[i] = ub[i]
            short[i] = ub[i]
            long[i] = np.nan
    return trend, direction, long, short


# ----------------------------------------------------------------------
# Public Supertrend function using Numba + our ATR
# ----------------------------------------------------------------------
def supertrend_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 7,
    atr_length: int | None = None,
    multiplier: float = 3.0,
    atr_mamode: str = "rma",
    offset: int = 0,
    fillna: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supertrend using Numba and our own ATR (Numba version).

    Returns (trend, direction, long, short) as numpy arrays.
    """
    if atr_length is None:
        atr_length = length

    # Ensure contiguous
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # Midpoint (hl2)
    hl2 = (high + low) * 0.5
    # ATR using Numba
    atr = atr_ind(
        high, low, close, 
        length=atr_length, 
        mamode=atr_mamode, 
        drift=1, 
        offset=0, 
        fillna=None, 
        percent=False,
        use_talib=False
    )
    # Bands
    ub = hl2 + multiplier * atr
    lb = hl2 - multiplier * atr
    # Core logic
    trend, direction, long, short = _supertrend_numba_core(close, lb, ub, initial_dir=1)
    # Apply final offset and fillna to each array
    trend = _apply_offset_fillna(trend, offset, fillna)
    direction = _apply_offset_fillna(direction, offset, fillna)
    long = _apply_offset_fillna(long, offset, fillna)
    short = _apply_offset_fillna(short, offset, fillna)

    return trend, direction, long, short


# ----------------------------------------------------------------------
# Version using TA‑Lib ATR if available and requested
# ----------------------------------------------------------------------
def supertrend_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 7,
    atr_length: int | None = None,
    multiplier: float = 3.0,
    offset: int = 0,
    fillna: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supertrend using TA‑Lib ATR (if available) and Numba core.
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    if atr_length is None:
        atr_length = length
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    hl2 = (high + low) * 0.5
    atr = atr_ind(
        high, low, close, 
        length=atr_length, 
        offset=0, 
        fillna=None, 
        percent=False, 
        use_talib=True
    )
    ub = hl2 + multiplier * atr
    lb = hl2 - multiplier * atr
    trend, direction, long, short = _supertrend_numba_core(close, lb, ub, initial_dir=1)
    trend = _apply_offset_fillna(trend, offset, fillna)
    direction = _apply_offset_fillna(direction, offset, fillna)
    long = _apply_offset_fillna(long, offset, fillna)
    short = _apply_offset_fillna(short, offset, fillna)
    return trend, direction, long, short


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def supertrend_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 7,
    atr_length: int | None = None,
    multiplier: float = 3.0,
    atr_mamode: str = "rma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal Supertrend with backend selection.

    Parameters
    ----------
    high, low, close : np.ndarray or pl.Series
        Price series.
    length : int
        Main period (used for ATR if atr_length not given).
    atr_length : int, optional
        ATR period (defaults to length).
    multiplier : float
        Band multiplier.
    atr_mamode : str
        MA type for ATR (ignored if use_talib=True).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA‑Lib for ATR if available.

    Returns
    -------
    tuple of np.ndarray
        (trend, direction, long, short) as numpy arrays.
    """
    # Convert Polars to numpy
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return supertrend_talib(
            high, low, close, length, atr_length, multiplier, offset, fillna
        )
    else:
        return supertrend_numba(
            high, low, close, length, atr_length, multiplier, atr_mamode, offset, fillna
        )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def supertrend_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    length: int = 7,
    atr_length: int | None = None,
    multiplier: float = 3.0,
    atr_mamode: str = "rma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = ""
) -> pl.DataFrame:
    """
    Add Supertrend columns to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Column names for prices.
    length, atr_length, multiplier, atr_mamode, offset, fillna, use_talib : as above.
    suffix : str
        Suffix for column names (default f"_{length}_{multiplier}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with columns:
        SUPERT{suffix}, SUPERTd{suffix}, SUPERTl{suffix}, SUPERTs{suffix}
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    trend, direction, long, short = supertrend_ind(
        high, low, close,
        length=length,
        atr_length=atr_length,
        multiplier=multiplier,
        atr_mamode=atr_mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib
    )

    suffix = suffix or f"_{length}_{multiplier}"
    return df.with_columns([
        pl.Series(f"SUPERT{suffix}", trend),
        pl.Series(f"SUPERTd{suffix}", direction),
        pl.Series(f"SUPERTl{suffix}", long),
        pl.Series(f"SUPERTs{suffix}", short)
    ])