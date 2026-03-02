# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core Numba implementation (Ehlers' MAMA) – final optimized version
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _mama_numba_core(
    close: np.ndarray,
    fastlimit: float,
    slowlimit: float,
    prenan: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAMA core loop (Numba). Returns (mama, fama).
    """
    n = len(close)
    mama = np.full(n, np.nan, dtype=np.float64)
    fama = np.full(n, np.nan, dtype=np.float64)
    if n < 6:
        return mama, fama
    a, b = 0.0962, 0.5769
    p_w = 0.2
    # Temporary arrays
    wma4 = np.empty(n, dtype=np.float64)
    dt = np.empty(n, dtype=np.float64)
    i1 = np.empty(n, dtype=np.float64)
    i2 = np.empty(n, dtype=np.float64)
    q1 = np.empty(n, dtype=np.float64)
    q2 = np.empty(n, dtype=np.float64)
    ji = np.empty(n, dtype=np.float64)
    jq = np.empty(n, dtype=np.float64)
    re = np.empty(n, dtype=np.float64)
    im = np.empty(n, dtype=np.float64)
    period = np.empty(n, dtype=np.float64)
    phase = np.empty(n, dtype=np.float64)
    alpha = np.empty(n, dtype=np.float64)

    # Initialise first 6 values
    for i in range(6):
        wma4[i] = dt[i] = \
            i1[i] = i2[i] = q1[i] = q2[i] = \
                ji[i] = jq[i] = re[i] = im[i] = 0.0
        period[i] = 0.0
        phase[i] = 0.0
        alpha[i] = 0.0
        mama[i] = close[i]
        fama[i] = close[i]
    for i in range(6, n):
        c = close[i]
        c1 = close[i - 1]
        c2 = close[i - 2]
        c3 = close[i - 3]
        adj_prev_period = 0.075 * period[i - 1] + 0.54
        wma4[i] = 0.4 * c + 0.3 * c1 + 0.2 * c2 + 0.1 * c3
        dt[i] = adj_prev_period * (
            a * wma4[i] + b * wma4[i - 2] - b * wma4[i - 4] - a * wma4[i - 6]
        )
        q1[i] = adj_prev_period * (
            a * dt[i] + b * dt[i - 2] - b * dt[i - 4] - a * dt[i - 6]
        )
        i1[i] = dt[i - 3]
        ji[i] = adj_prev_period * (
            a * i1[i] + b * i1[i - 2] - b * i1[i - 4] - a * i1[i - 6]
        )
        jq[i] = adj_prev_period * (
            a * q1[i] + b * q1[i - 2] - b * q1[i - 4] - a * q1[i - 6]
        )
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]
        i2[i] = p_w * i2[i] + (1 - p_w) * i2[i - 1]
        q2[i] = p_w * q2[i] + (1 - p_w) * q2[i - 1]
        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] + q2[i] * i2[i - 1]
        re[i] = p_w * re[i] + (1 - p_w) * re[i - 1]
        im[i] = p_w * im[i] + (1 - p_w) * im[i - 1]
        if im[i] != 0.0 and re[i] != 0.0:
            period[i] = 360.0 / np.arctan(im[i] / re[i])
        else:
            period[i] = 0.0
        # Period limits
        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6.0:
            period[i] = 6.0
        if period[i] > 50.0:
            period[i] = 50.0
        period[i] = p_w * period[i] + (1 - p_w) * period[i - 1]
        if i1[i] != 0.0:
            phase[i] = np.arctan(q1[i] / i1[i])
        else:
            phase[i] = phase[i - 1]
        dphase = phase[i - 1] - phase[i]
        if dphase < 1.0:
            dphase = 1.0
        alpha[i] = fastlimit / dphase
        if alpha[i] > fastlimit:
            alpha[i] = fastlimit
        if alpha[i] < slowlimit:
            alpha[i] = slowlimit
        mama[i] = alpha[i] * c + (1 - alpha[i]) * mama[i - 1]
        fama[i] = 0.5 * alpha[i] * mama[i] + (1 - 0.5 * alpha[i]) * fama[i - 1]
    if prenan > 0:
        mama[:prenan] = np.nan
        fama[:prenan] = np.nan
    return mama, fama


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def mama_numba(
    close: np.ndarray,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    prenan: int = 3,
    offset: int = 0,
    fillna: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAMA using Numba. Returns (mama, fama).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    mama, fama = _mama_numba_core(close, fastlimit, slowlimit, prenan)
    # Apply offset/fillna to each array
    mama = _apply_offset_fillna(mama, offset, fillna)
    fama = _apply_offset_fillna(fama, offset, fillna)
    return mama, fama


# ----------------------------------------------------------------------
# TA‑Lib wrapper
# ----------------------------------------------------------------------
def mama_talib(
    close: np.ndarray,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    offset: int = 0,
    fillna: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAMA using TA‑Lib. Returns (mama, fama).
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    mama, fama = talib.MAMA(close, fastlimit=fastlimit, slowlimit=slowlimit)
    mama = _apply_offset_fillna(mama, offset, fillna)
    fama = _apply_offset_fillna(fama, offset, fillna)
    return mama, fama


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def mama_ind(
    close: np.ndarray | pl.Series,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    prenan: int = 3,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal MAMA with backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    fastlimit, slowlimit : float
        Limits for adaptive alpha.
    prenan : int
        Number of initial NaN values (Numba only).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA‑Lib if available.

    Returns
    -------
    (mama, fama) : tuple of np.ndarray
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    if use_talib and talib_available:
        return mama_talib(close, fastlimit, slowlimit, offset, fillna)
    else:
        return mama_numba(close, fastlimit, slowlimit, prenan, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def mama_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    prenan: int = 3,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    suffix: str = ""
) -> pl.DataFrame:
    """
    Add MAMA and FAMA columns to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    fastlimit, slowlimit, prenan, offset, fillna, use_talib : as above.
    suffix : str
        Suffix for output columns (default f"_{fastlimit}_{slowlimit}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with columns 'MAMA{suffix}' and 'FAMA{suffix}'.
    """
    close = df[close_col].to_numpy()
    mama_arr, fama_arr = mama_ind(
        close, fastlimit, slowlimit, prenan, offset, fillna, use_talib
    )
    suffix = suffix or f"_{fastlimit}_{slowlimit}"
    return df.with_columns([
        pl.Series(f"MAMA{suffix}", mama_arr),
        pl.Series(f"FAMA{suffix}", fama_arr)
    ])