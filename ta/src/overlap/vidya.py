# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# CMO (Chande Momentum Oscillator) – Numba version
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _cmo_numba(close: np.ndarray, length: int, drift: int) -> np.ndarray:
    """
    Compute Chande Momentum Oscillator (CMO) using Numba.
    Returns an array of CMO values, same length as `close`.
    First (length + drift - 1) values are NaN.
    """
    n = len(close)
    cmo = np.full(n, np.nan, dtype=np.float64)
    # Need at least length+drift to compute anything
    if n < length + drift:
        return cmo
    # We'll compute rolling sums of positive and negative changes
    # Pre‑compute differences
    diff = np.empty(n, dtype=np.float64)
    for i in range(drift, n):
        diff[i] = close[i] - close[i - drift]
    # Rolling sums using cumulative sum trick
    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)
    # Cumulative sums
    cum_pos = np.zeros(n + 1, dtype=np.float64)
    cum_neg = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        cum_pos[i] = cum_pos[i - 1] + pos[i - 1]
        cum_neg[i] = cum_neg[i - 1] + neg[i - 1]
    # Rolling sums over window `length`
    for i in range(length, n):
        pos_sum = cum_pos[i] - cum_pos[i - length]
        neg_sum = cum_neg[i] - cum_neg[i - length]
        denom = pos_sum + neg_sum
        if denom != 0.0:
            cmo[i] = (pos_sum - neg_sum) / denom
        else:
            cmo[i] = 0.0  # avoid division by zero
    return cmo


# ----------------------------------------------------------------------
# VIDYA using Numba
# ----------------------------------------------------------------------
def vidya_numba(
    close: np.ndarray,
    length: int = 14,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    VIDYA using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    n = len(close)
    # alpha = 2 / (length + 1)
    alpha = 2.0 / (length + 1)
    # Compute CMO
    cmo = _cmo_numba(close, length, drift)
    abs_cmo = np.abs(cmo)
    # Initialize VIDYA array
    vidya = np.full(n, 0.0, dtype=np.float64)
    # First value is 0 (will be replaced by NaN later)
    for i in range(1, n):
        # VIDYA formula
        sc = alpha * abs_cmo[i]
        vidya[i] = sc * close[i] + (1.0 - sc) * vidya[i - 1]
    # Set first `length` values to NaN (original behaviour: 0 -> nan)
    # Also any remaining zeros from the first few will be replaced.
    for i in range(min(length, n)):
        vidya[i] = np.nan
    return _apply_offset_fillna(vidya, offset, fillna)


# ----------------------------------------------------------------------
# VIDYA using TA‑Lib (if available)
# ----------------------------------------------------------------------
def vidya_talib(
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    VIDYA using TA‑Lib (C implementation). Note: TA‑Lib provides CMO,
    but not VIDYA directly. We'll compute CMO and then apply VIDYA formula
    in Python (non‑Numba). For consistency, we still return numpy array.
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    n = len(close)
    alpha = 2.0 / (length + 1)
    # Compute CMO via TA‑Lib
    cmo = talib.CMO(close, timeperiod=length)
    abs_cmo = np.abs(cmo)
    # Python loop for VIDYA (can't use Numba here, but TA‑Lib part is fast)
    vidya = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        sc = alpha * abs_cmo[i]
        vidya[i] = sc * close[i] + (1.0 - sc) * vidya[i - 1]
    # Set first `length` to NaN
    for i in range(min(length, n)):
        vidya[i] = np.nan
    return _apply_offset_fillna(vidya, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def vidya_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal VIDYA with backend selection.
    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Period for CMO and alpha.
    drift : int
        Shift for price differences (used only in Numba version).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        If True and TA‑Lib is available, use it for CMO (VIDYA loop still in Python).

    Returns
    -------
    np.ndarray
        VIDYA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return vidya_talib(close, length, offset, fillna)
    else:
        return vidya_numba(close, length, drift, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def vidya_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    Add VIDYA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, drift, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"VIDYA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with VIDYA column.
    """
    close = df[close_col].to_numpy()
    result = vidya_ind(close, length, drift, offset, fillna, use_talib)
    out_name = output_col or f"VIDYA_{length}"
    return df.with_columns([pl.Series(out_name, result)])