# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, njit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (float64[:], float64[:], float64[:], float64[:]),
    nopython=True,
    cache=True
)
def _cdl_kicking_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba‑accelerated Kicking pattern.
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
        h2 = high[i]
        l2 = low[i]
        rng1 = h1 - l1
        rng2 = h2 - l2
        if rng1 <= 0.0 or rng2 <= 0.0:
            continue
        # Marubozu approximation: very small shadows
        upper1 = h1 - max(o1, c1)
        lower1 = min(o1, c1) - l1
        upper2 = h2 - max(o2, c2)
        lower2 = min(o2, c2) - l2
        # First candle marubozu
        if upper1 > 0.1 * rng1 or lower1 > 0.1 * rng1:
            continue
        # Second candle marubozu
        if upper2 > 0.1 * rng2 or lower2 > 0.1 * rng2:
            continue
        # Opposite colors
        bull1 = c1 > o1
        bear1 = c1 < o1
        bull2 = c2 > o2
        bear2 = c2 < o2
        if not ((bull1 and bear2) or (bear1 and bull2)):
            continue
        # Bodies non-overlapping (gap between bodies)
        body1_high = max(o1, c1)
        body1_low = min(o1, c1)
        body2_high = max(o2, c2)
        body2_low = min(o2, c2)
        # Either body2 entirely above body1, or entirely below
        if not (body2_low > body1_high or body2_high < body1_low):
            continue
        out[i] = True
    return out


def cdl_kicking(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Kicking pattern.
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
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    if use_talib and talib_available:
        talib_out = talib.CDLKICKING(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    mask = _cdl_kicking_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_kicking_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_KICKING",
) -> pl.DataFrame:
    """
    Add Kicking column to Polars DataFrame.
    """
    out = cdl_kicking(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))

