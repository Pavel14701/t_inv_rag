# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, njit

from ..external import talib, talib_available
from .._array_ops import _apply_offset_fillna


@njit(
    (float64[:], float64[:], float64[:], float64[:]),
    cache=True
)
def _cdl_identical3crows_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Numba‑accelerated Identical Three Crows pattern.
    Returns boolean mask where pattern completes (True at the third candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # Candles i-2, i-1, i must be black (bearish)
        if not (close[i - 2] < open_[i - 2]):
            continue
        if not (close[i - 1] < open_[i - 1]):
            continue
        if not (close[i] < open_[i]):
            continue
        # Each close lower than previous close (downward progression)
        if not (close[i - 1] < close[i - 2]):
            continue
        if not (close[i] < close[i - 1]):
            continue
        # Small lower shadows (closes near lows)
        rng0 = high[i - 2] - low[i - 2]
        rng1 = high[i - 1] - low[i - 1]
        rng2 = high[i] - low[i]
        if rng0 <= 0.0 or rng1 <= 0.0 or rng2 <= 0.0:
            continue
        low_sh0 = close[i - 2] - low[i - 2]
        low_sh1 = close[i - 1] - low[i - 1]
        low_sh2 = close[i] - low[i]
        if low_sh0 > 0.25 * rng0:
            continue
        if low_sh1 > 0.25 * rng1:
            continue
        if low_sh2 > 0.25 * rng2:
            continue
        # Opens near previous closes (identical crows)
        if abs(open_[i - 1] - close[i - 2]) > 0.25 * rng1:
            continue
        if abs(open_[i] - close[i - 1]) > 0.25 * rng2:
            continue
        out[i] = True
    return out


def cdl_identical3crows(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Identical Three Crows pattern.
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
        talib_out = talib.CDLIDENTICAL3CROWS(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    mask = _cdl_identical3crows_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_identical3crows_polars(
    df: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = 'CDL_IDENTICAL3CROWS',
) -> pl.DataFrame:
    """Add Identical Three Crows column to Polars DataFrame."""
    out = cdl_identical3crows(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
