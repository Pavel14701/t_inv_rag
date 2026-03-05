# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _quantile_numba_core(close: np.ndarray, length: int, q: float) -> np.ndarray:
    """
    Rolling quantile using a sliding window and sorting each window.
    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.
    q : float
        Quantile (0 < q < 1).
    Returns
    -------
    np.ndarray
        Quantile values; first `length-1` positions are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1].copy()  # need a copy for sorting
        window.sort()
        idx = int(round(q * (length - 1)))  # typical quantile interpolation
        out[i] = window[idx]
    return out


def quantile_numba(
    close: np.ndarray,
    length: int = 30,
    q: float = 0.5,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Rolling quantile using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _quantile_numba_core(close, length, q)
    return _apply_offset_fillna(result, offset, fillna)


def quantile_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    q: float = 0.5,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal rolling quantile (always uses Numba).
    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Window length.
    q : float
        Quantile (0 < q < 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        Quantile values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return quantile_numba(close, length, q, offset, fillna)


def quantile_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    q: float = 0.5,
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
    length, q, offset, fillna : as above.
    output_col : str, optional
        Output column name (default f"QTL_{length}_{q}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    result = quantile_ind(close, length, q, offset, fillna)
    out_name = output_col or f"QTL_{length}_{q}"
    return pl.Series(out_name, result)