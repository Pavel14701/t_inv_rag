# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _brar_numba_core(
    high_open_range: np.ndarray,
    open_low_range: np.ndarray,
    hcy: np.ndarray,
    cyl: np.ndarray,
    length: int,
    scalar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute AR and BR using sliding window sums (Numba).
    """
    n = len(high_open_range)
    ar = np.full(n, np.nan, dtype=np.float64)
    br = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return ar, br
    # Initial sums
    sum_high_open = 0.0
    sum_open_low = 0.0
    sum_hcy = 0.0
    sum_cyl = 0.0
    for i in range(length):
        sum_high_open += high_open_range[i]
        sum_open_low += open_low_range[i]
        sum_hcy += hcy[i]
        sum_cyl += cyl[i]
    if sum_open_low != 0.0:
        ar[length - 1] = scalar * sum_high_open / sum_open_low
    if sum_cyl != 0.0:
        br[length - 1] = scalar * sum_hcy / sum_cyl
    for i in range(length, n):
        j = i - length
        sum_high_open += high_open_range[i] - high_open_range[j]
        sum_open_low += open_low_range[i] - open_low_range[j]
        sum_hcy += hcy[i] - hcy[j]
        sum_cyl += cyl[i] - cyl[j]
        if sum_open_low != 0.0:
            ar[i] = scalar * sum_high_open / sum_open_low
        if sum_cyl != 0.0:
            br[i] = scalar * sum_hcy / sum_cyl
    return ar, br


def brar_ind(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 26,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal BRAR indicator (always uses Numba).

    Parameters
    ----------
    open_, high, low, close : np.ndarray or pl.Series
        OHLC price series.
    length : int
        Window length.
    scalar : float
        Multiplier.
    drift : int
        Shift for close.
    offset, fillna : as usual.

    Returns
    -------
    ar, br : tuple of np.ndarray
    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    # Ensure float64 contiguous
    open_ = np.asarray(open_, dtype=np.float64, copy=False)
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # Compute ranges
    high_open_range = high - open_
    open_low_range = open_ - low
    # Shifted close
    close_shifted = np.roll(close, drift)
    close_shifted[:drift] = np.nan
    hcy = np.maximum(high - close_shifted, 0.0)
    cyl = np.maximum(close_shifted - low, 0.0)
    ar, br = _brar_numba_core(
        high_open_range, open_low_range, hcy, cyl, length, scalar
    )
    ar = _apply_offset_fillna(ar, offset, fillna)
    br = _apply_offset_fillna(br, offset, fillna)
    return ar, br


def brar_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    length: int = 26,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Returns DataFrame with date  AR, BR columns.
    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    ar, br = brar_ind(open_arr, high_arr, low_arr, close_arr,
                  length=length, scalar=scalar, drift=drift,
                  offset=offset, fillna=fillna)
    suffix = suffix or f"_{length}"
    return pl.DataFrame({
        date_col: df[date_col],
        f"AR{suffix}": ar,
        f"BR{suffix}": br,
    })