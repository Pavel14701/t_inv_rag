# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from numba import float64, njit

from .._array_ops import _apply_offset_fillna
from ..external import talib, talib_available


@njit((float64[:], float64[:], float64[:], float64[:]), cache=True)
def _cdl_inneck_nb(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Numba-accelerated In-Neck pattern.
    Returns boolean mask where pattern completes (True at the second candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        o1 = open_[i - 1]
        c1 = close[i - 1]
        h1 = high[i - 1]
        l1 = low[i - 1]
        o2 = open_[i]
        c2 = close[i]
        high[i]
        low[i]
        # First candle: long black (bearish)
        if not (c1 < o1):
            continue
        rng1 = h1 - l1
        if rng1 <= 0.0:
            continue
        # Second candle: white (bullish)
        if not (c2 > o2):
            continue
        # Gap down open: second opens below first low
        if not (o2 < l1):
            continue
        # Close of second near close of first, but not above first open
        if not (c2 > c1 and c2 <= o1):
            continue
        # "In-neck": close very near previous close
        if (c2 - c1) > 0.25 * rng1:
            continue
        out[i] = True
    return out


def cdl_inneck(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal In-Neck pattern.
    Returns numpy array of float64: 1.0 where pattern occurs, else 0.0.
    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not open_.flags.writeable:
        open_ = open_.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not low.flags.writeable:
        low = low.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    if use_talib and talib_available:
        talib_out = talib.CDLINNECK(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    mask = _cdl_inneck_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_inneck_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_INNECK",
) -> pl.DataFrame:
    """Add In-Neck column to Polars DataFrame."""
    out = cdl_inneck(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
