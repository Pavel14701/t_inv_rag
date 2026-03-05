# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from ..ma import ma_mode
from ..utils import _apply_offset_fillna
from .atr import atr_ind


@jit(nopython=True, fastmath=True, cache=True)
def _atrts_numba_core(
    close: np.ndarray,
    ma: np.ndarray,
    atr: np.ndarray,
    length: int,
    ma_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core ATR Trailing Stop logic.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    ma : np.ndarray
        Moving average of close (same length).
    atr : np.ndarray
        ATR values multiplied by k (same length).
    length : int
        ATR period (used only to determine initial NaN region).
    ma_length : int
        MA period (used to determine initial NaN region).

    Returns
    -------
    atrts, long_stop, short_stop : np.ndarray
        Main trailing stop line, long stop values, short stop values.
        First max(length, ma_length) values are NaN.
    """
    n = len(close)
    atrts = np.empty(n, dtype=np.float64)
    long_stop = np.empty(n, dtype=np.float64)
    short_stop = np.empty(n, dtype=np.float64)
    k = max(length, ma_length)
    for i in range(k):
        atrts[i] = np.nan
        long_stop[i] = np.nan
        short_stop[i] = np.nan
    if n <= k:
        return atrts, long_stop, short_stop
    up = close[k] > ma[k]
    if up:
        atrts[k] = close[k] - atr[k]
        long_stop[k] = atrts[k]
        short_stop[k] = np.nan
    else:
        atrts[k] = close[k] + atr[k]
        short_stop[k] = atrts[k]
        long_stop[k] = np.nan
    for i in range(k + 1, n):
        up_now = close[i] > ma[i]
        if up_now:
            new_stop = close[i] - atr[i]
            new_stop = max(new_stop, atrts[i - 1])
            atrts[i] = new_stop
            long_stop[i] = new_stop
            short_stop[i] = np.nan
        else:
            new_stop = close[i] + atr[i]
            new_stop = min(new_stop, atrts[i - 1])
            atrts[i] = new_stop
            short_stop[i] = new_stop
            long_stop[i] = np.nan
    return atrts, long_stop, short_stop


def atrts_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    ma_length: int = 20,
    k: float = 3.0,
    mamode: str = "ema",
    drift: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    percent: bool = False,
) -> np.ndarray:
    """
    Numpy‑based ATR Trailing Stop.

    Returns main ATRTS line as numpy array.
    """
    # Input preparation
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

    # Compute ATR (using existing atr_ind)
    atr = atr_ind(
        high, low, close,
        length=length,
        mamode=mamode,
        drift=drift,
        offset=0,
        fillna=None,
        percent=False,
        use_talib=use_talib,
    )
    atr = atr * k
    _ma = ma_mode(
        source=close,
        length=ma_length,
        mamode=mamode,
        offset=0,
        fillna=None,
        use_talib=use_talib,
    )
    atrts, _, _ = _atrts_numba_core(close, _ma, atr, length, ma_length)
    if percent:
        atrts = atrts * 100.0 / close
    return _apply_offset_fillna(atrts, offset, fillna)


def atrts(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    ma_length: int = 20,
    k: float = 3.0,
    mamode: str = "ema",
    drift: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    percent: bool = False,
) -> np.ndarray:
    """
    Universal ATR Trailing Stop (accepts numpy arrays or Polars Series).

    Returns main ATRTS line as numpy array.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return atrts_numpy(
        high, low, close,
        length=length,
        ma_length=ma_length,
        k=k,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        percent=percent,
    )


def atrts_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    length: int = 14,
    ma_length: int = 20,
    k: float = 3.0,
    mamode: str = "ema",
    drift: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    percent: bool = False,
    output_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Add ATR Trailing Stop column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Column names for prices.
    length, ma_length, k, mamode, drift, offset, fillna, use_talib, percent : as above.
    output_col : str, optional
        Output column name (default f"ATRTS_{length}_{ma_length}_{k}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with ATRTS column.
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = atrts_numpy(
        high, low, close,
        length=length,
        ma_length=ma_length,
        k=k,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        percent=percent,
    )
    out_name = output_col or f"ATRTS_{length}_{ma_length}_{k}"
    return df.with_columns([pl.Series(out_name, result)])