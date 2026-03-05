# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _entropy_numba_core(close: np.ndarray, length: int, base: float) -> np.ndarray:
    """
    Rolling entropy via sliding window (Numba).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.
    base : float
        Logarithm base (e.g., 2.0 for bits).

    Returns
    -------
    np.ndarray
        Entropy values; first `length-1` positions are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    log_base = np.log(base)  # pre‑compute denominator once
    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        s = window.sum()
        if s <= 0.0:
            # if sum is zero, entropy is undefined (NaN)
            out[i] = np.nan
            continue
        entropy = 0.0
        for j in range(length):
            p = window[j] / s
            if p > 0.0:
                entropy -= p * np.log(p)
        out[i] = entropy / log_base
    return out


def entropy_numba(
    close: np.ndarray,
    length: int = 10,
    base: float = 2.0,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Rolling entropy using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _entropy_numba_core(close, length, base)
    return _apply_offset_fillna(result, offset, fillna)


def entropy_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    base: float = 2.0,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal rolling entropy (always uses Numba).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Window length.
    base : float
        Logarithm base (e.g., 2.0 for bits).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        Entropy values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return entropy_numba(close, length, base, offset, fillna)


def entropy_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    base: float = 2.0,
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
    length, base, offset, fillna : as above.
    output_col : str, optional
        Output column name (default f"ENTP_{length}").

    Returns
    -------
    pl.Series
    """
    close = df[close_col].to_numpy()
    result = entropy_ind(close, length, base, offset, fillna)
    out_name = output_col or f"ENTP_{length}"
    return pl.Series(out_name, result)