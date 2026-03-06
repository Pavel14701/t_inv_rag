# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@jit((float64[:], int64, float64), nopython=True, fastmath=True, cache=True)
def _roc_numba_core(close: np.ndarray, length: int, scalar: float) -> np.ndarray:
    """
    Rate of Change core calculation.
    ROC = scalar * (close[i] - close[i-length]) / close[i-length]
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length + 1:
        return out
    for i in range(length, n):
        numerator = close[i] - close[i - length]
        denominator = close[i - length]
        if denominator != 0.0:
            out[i] = scalar * numerator / denominator
        else:
            out[i] = np.nan  # division by zero
    return out


def roc_numpy(
    close: np.ndarray,
    length: int = 10,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based ROC calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        ROC period.
    scalar : float
        Multiplier (e.g., 100 for percent).
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        ROC values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if use_talib and talib_available:
        # TA‑Lib ROC returns percentage (scalar=100)
        result = talib.ROC(close, timeperiod=length)
        if scalar != 100.0:
            result = result * (scalar / 100.0)
    else:
        result = _roc_numba_core(close, length, scalar)
    return _apply_offset_fillna(result, offset, fillna)


def roc_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal ROC (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return roc_numpy(close, length, scalar, offset, fillna, use_talib)


def roc_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    length: int = 10,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add ROC column and return a new DataFrame with date and ROC.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    date_col : str
        Column with dates.
    length, scalar, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"ROC_{length}").

    Returns
    -------
    pl.DataFrame
        DataFrame with date and ROC columns.
    """
    close = df[close_col].to_numpy()
    result = roc_numpy(close, length, scalar, offset, fillna, use_talib)
    out_name = output_col or f"ROC_{length}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result,
    })