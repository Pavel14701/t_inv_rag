# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import polars as pl
from numba import float64, njit

from ..utils import _apply_offset_fillna


@njit((float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def _np_ha(np_open, np_high, np_low, np_close):
    """
    Numba‑compiled core for Heikin-Ashi calculation.
    """
    ha_close = 0.25 * (np_open + np_high + np_low + np_close)
    ha_open = np.empty_like(ha_close)
    ha_open[0] = 0.5 * (np_open[0] + np_close[0])
    m = np_close.size
    for i in range(1, m):
        ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close[i - 1])
    ha_high = np.maximum(np.maximum(ha_open, ha_close), np_high)
    ha_low = np.minimum(np.minimum(ha_open, ha_close), np_low)
    return ha_open, ha_high, ha_low, ha_close


def ha_numpy(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based Heikin-Ashi calculation.

    Returns (ha_open, ha_high, ha_low, ha_close) as numpy arrays.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    # Ensure contiguous
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    ha_open, ha_high, ha_low, ha_close = _np_ha(open_, high, low, close)
    # Apply offset and fillna
    ha_open = _apply_offset_fillna(ha_open, offset, fillna)
    ha_high = _apply_offset_fillna(ha_high, offset, fillna)
    ha_low = _apply_offset_fillna(ha_low, offset, fillna)
    ha_close = _apply_offset_fillna(ha_close, offset, fillna)
    return ha_open, ha_high, ha_low, ha_close


def ha(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal Heikin-Ashi (accepts numpy arrays or Polars Series).
    Returns (ha_open, ha_high, ha_low, ha_close) as numpy arrays.
    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ha_numpy(open_, high, low, close, offset, fillna)


def ha_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    offset: int = 0,
    fillna: Optional[float] = None,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Return a new Polars DataFrame with Heikin-Ashi candles.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    open_col, high_col, low_col, close_col : str
        Names of the columns with OHLC prices.
    date_col : str
        Name of the date/time column (will be included in the output).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs after shifting.
    suffix : str
        Suffix for output columns (default "").

    Returns
    -------
    pl.DataFrame
        New DataFrame with columns:
            date_col,
            HA_open{suffix}, HA_high{suffix}, HA_low{suffix}, HA_close{suffix}.
    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    ha_open, ha_high, ha_low, ha_close = ha_numpy(
        open_arr, high_arr, low_arr, close_arr, offset, fillna
    )
    suffix = suffix or ""
    return pl.DataFrame({
        date_col: df[date_col],
        f"HA_open{suffix}": ha_open,
        f"HA_high{suffix}": ha_high,
        f"HA_low{suffix}": ha_low,
        f"HA_close{suffix}": ha_close,
    })