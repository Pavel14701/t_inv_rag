# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _kurtosis_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """
    Rolling kurtosis (Fisher's) using sliding sums of powers (Numba).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length (must be at least 4 to obtain a finite value).

    Returns
    -------
    np.ndarray
        Kurtosis values; first `length-1` positions are NaN.
        For windows with length < 4, the result is NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial window sums
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for i in range(length):
        x = close[i]
        s1 += x
        s2 += x * x
        s3 += x * x * x
        s4 += x * x * x * x

    def compute_kurtosis(s1, s2, s3, s4, L):
        if L < 4:
            return np.nan
        mean = s1 / L
        # central moments (not normalized)
        M2 = s2 - L * mean * mean
        M4 = s4 - 4.0 * mean * s3 + 6.0 * mean * mean * s2 \
            - 3.0 * L * mean * mean * mean * mean
        if M2 <= 0.0:
            return np.nan
        # Fisher's kurtosis (excess kurtosis, unbiased for normal)
        kurt = (L * (L + 1) * M4 - 3.0 * (L - 1) * M2 * M2) \
            / ((L - 2) * (L - 3) * M2 * M2)
        return kurt
    out[length - 1] = compute_kurtosis(s1, s2, s3, s4, length)
    # Rolling update
    for i in range(length, n):
        add = close[i]
        rem = close[i - length]
        s1 += add - rem
        s2 += add * add - rem * rem
        s3 += add * add * add - rem * rem * rem
        s4 += add * add * add * add - rem * rem * rem * rem
        out[i] = compute_kurtosis(s1, s2, s3, s4, length)
    return out


def kurtosis_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Rolling kurtosis using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _kurtosis_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def kurtosis_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal rolling kurtosis (always uses Numba).

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
        Kurtosis values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return kurtosis_numba(close, length, offset, fillna)


def kurtosis_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
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
        Output column name (default f"KURT_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    result = kurtosis_ind(close, length, offset, fillna)
    out_name = output_col or f"KURT_{length}"
    return df.with_columns([pl.Series(out_name, result)])