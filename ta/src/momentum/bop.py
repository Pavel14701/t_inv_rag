# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


def bop_numpy(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    scalar: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Numpy‑based Balance of Power 
    (used when TA‑Lib is not available or disabled).

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC arrays (float64).
    scalar : float
        Multiplier.
    offset, fillna : as usual.

    Returns
    -------
    np.ndarray
        BOP values.
    """
    # Ensure contiguous
    open_ = np.asarray(open_, dtype=np.float64, copy=False)
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    hl_range = high - low
    co_range = close - open_
    # Avoid division by zero – where hl_range == 0, 
    # bop becomes inf. We'll keep as is (original behaviour).
    bop = scalar * co_range / hl_range
    return _apply_offset_fillna(bop, offset, fillna)


def bop_talib(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    TA‑Lib based Balance of Power (scalar is ignored).
    """
    if not talib_available:
        raise ImportError("TA‑Lib not available")
    open_ = np.asarray(open_, dtype=np.float64, copy=False)
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    bop = talib.BOP(open_, high, low, close)
    return _apply_offset_fillna(bop, offset, fillna)


def bop_ind(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    scalar: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Balance of Power.

    Parameters
    ----------
    open_, high, low, close : np.ndarray or pl.Series
        OHLC price series.
    scalar : float
        Multiplier (ignored if use_talib=True).
    offset, fillna : as usual.
    use_talib : bool
        If True and TA‑Lib is available, use it; else use Numpy version.

    Returns
    -------
    np.ndarray
        BOP values.
    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return bop_talib(open_, high, low, close, offset, fillna)
    else:
        return bop_numpy(open_, high, low, close, scalar, offset, fillna)


def bop_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    scalar: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str = "BOP",
) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    open_col, high_col, low_col, close_col : str
        Column names for OHLC.
    scalar, offset, fillna, use_talib : as above.
    output_col : str
        Output column name (default "BOP").

    Returns
    -------
    pl.DataFrame
    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    result = bop_ind(
        open_arr, high_arr, low_arr, close_arr, scalar, offset, fillna, use_talib
    )
    return pl.DataFrame({
        date_col: df[date_col],
        output_col: result
    })