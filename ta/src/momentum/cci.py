# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..overlap.hlc3 import hlc3_ind
from ..overlap.sma import sma_ind
from ..statistics.mad import mad_ind
from ..utils import _apply_offset_fillna


def cci_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    c: float = 0.015,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based CCI calculation.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64).
    length : int
        Period.
    c : float
        Scaling constant (ignored if use_talib=True).
    offset, fillna : as usual.
    use_talib : bool
        If True and TA‑Lib is available, use talib.CCI (c is ignored).

    Returns
    -------
    np.ndarray
        CCI values.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    if use_talib and talib_available:
        # TA‑Lib CCI uses fixed c=0.015, parameter not exposed
        result = talib.CCI(high, low, close, timeperiod=length)
    else:
        # Typical price
        tp = hlc3_ind(high, low, close, fillna=fillna)
        # SMA of typical price
        mean_tp = sma_ind(tp, length=length, offset=0, fillna=None, use_talib=False)
        # Mean absolute deviation of typical price
        mad_tp = mad_ind(tp, length=length, offset=0, fillna=None)
        # CCI formula: (TP - SMA(TP)) / (c * MAD(TP))
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (tp - mean_tp) / (c * mad_tp)
    return _apply_offset_fillna(result, offset, fillna)


def cci_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    c: float = 0.015,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal CCI (accepts numpy arrays or Polars Series).
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return cci_numpy(high, low, close, length, c, offset, fillna, use_talib)


def cci_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    length: int = 14,
    c: float = 0.015,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Column names for prices.
    length, c, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"CCI_{length}_{c}").

    Returns
    -------
    pl.DataFrame
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = cci_numpy(high, low, close, length, c, offset, fillna, use_talib)
    out_name = output_col or f"CCI_{length}_{c}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })