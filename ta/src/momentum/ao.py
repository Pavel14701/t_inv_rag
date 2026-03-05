# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap.sma import sma_ind
from ..utils import _apply_offset_fillna


def ao_numpy(
    high: np.ndarray,
    low: np.ndarray,
    fast: int = 5,
    slow: int = 34,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based Awesome Oscillator calculation.

    Parameters
    ----------
    high, low : np.ndarray
        Price arrays (float64).
    fast : int
        Fast SMA period.
    slow : int
        Slow SMA period.
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        AO values.
    """
    # Ensure arrays are contiguous
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    for arr in (high, low):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # Swap if slow < fast (original behaviour)
    if slow < fast:
        fast, slow = slow, fast
    median = (high + low) * 0.5
    fast_sma = sma_ind(median, length=fast, offset=0, fillna=None, use_talib=use_talib)
    slow_sma = sma_ind(median, length=slow, offset=0, fillna=None, use_talib=use_talib)
    ao = fast_sma - slow_sma
    return _apply_offset_fillna(ao, offset, fillna)


def ao_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    fast: int = 5,
    slow: int = 34,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Awesome Oscillator (accepts numpy arrays or Polars Series).
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    return ao_numpy(high, low, fast, slow, offset, fillna, use_talib)


def ao_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    date_col: str = "date",
    fast: int = 5,
    slow: int = 34,
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
    high_col, low_col : str
        Column names for high and low prices.
    fast, slow, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"AO_{fast}_{slow}").

    Returns
    -------
    pl.DataFrame
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    result = ao_numpy(high, low, fast, slow, offset, fillna, use_talib)
    out_name = output_col or f"AO_{fast}_{slow}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })