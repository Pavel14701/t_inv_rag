# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _skew_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """
    Rolling skewness using sliding sums of powers (Numba).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.
    Returns
    -------
    np.ndarray
        Skewness values; first `length-1` positions are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial window sums
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for i in range(length):
        x = close[i]
        sum1 += x
        sum2 += x * x
        sum3 += x * x * x

    # Helper to compute skew from sums
    def compute_skew(s1, s2, s3, L):
        if L < 3:
            return np.nan
        mean = s1 / L
        # central moments (not normalized)
        m2 = s2 - L * mean * mean
        m3 = s3 - 3.0 * mean * s2 + 2.0 * L * mean * mean * mean
        if m2 <= 0.0:
            return np.nan
        std = np.sqrt(m2 / (L - 1))  # sample standard deviation
        # Fisher‑Pearson skewness coefficient (bias‑corrected)
        skew = (L * m3) / ((L - 1) * (L - 2) * (std ** 3))
        return skew
    out[length - 1] = compute_skew(sum1, sum2, sum3, length)
    # Rolling update
    for i in range(length, n):
        add = close[i]
        rem = close[i - length]
        sum1 += add - rem
        sum2 += add * add - rem * rem
        sum3 += add * add * add - rem * rem * rem
        out[i] = compute_skew(sum1, sum2, sum3, length)
    return out


def skew_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Rolling skewness using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _skew_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def skew_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal rolling skewness (always uses Numba).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        Skewness values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return skew_numba(close, length, offset, fillna)


def skew_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """
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
    output_col : str, optional
        Output column name (default f"SKEW_{length}").

    Returns
    -------
    pl.Series
    """
    close = df[close_col].to_numpy()
    result = skew_ind(close, length, offset, fillna)
    out_name = output_col or f"SKEW_{length}"
    return pl.Series(out_name, result)