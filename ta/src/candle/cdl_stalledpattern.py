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
def _cdl_stalledpattern_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba-accelerated Stalled Pattern (Deliberation) pattern.
    Returns boolean mask where pattern completes (True at the 3rd candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # Candle 1: long white
        o1 = open_[i - 2]
        c1 = close[i - 2]
        h1 = high[i - 2]
        l1 = low[i - 2]
        if not (c1 > o1):
            continue
        rng1 = h1 - l1
        body1 = c1 - o1
        if rng1 <= 0.0 or body1 < 0.6 * rng1:
            continue  # long bullish
        # Candle 2: white, advancing
        o2 = open_[i - 1]
        c2 = close[i - 1]
        h2 = high[i - 1]
        l2 = low[i - 1]
        if not (c2 > o2):
            continue
        rng2 = h2 - l2
        body2 = c2 - o2
        if rng2 <= 0.0 or body2 < 0.5 * rng2:
            continue  # reasonably long bullish
        if not (c2 > c1):
            continue  # continuation up
        # Candle 3: small white, stalling, with upper shadow
        o3 = open_[i]
        c3 = close[i]
        h3 = high[i]
        l3 = low[i]
        if not (c3 > o3):
            continue
        rng3 = h3 - l3
        if rng3 <= 0.0:
            continue
        body3 = c3 - o3
        upper3 = h3 - c3
        lower3 = o3 - l3
        # body smaller than previous, not too big
        if body3 >= body2:
            continue
        if body3 > 0.5 * rng3:
            continue
        # noticeable upper shadow, small lower shadow
        if upper3 < 0.3 * rng3:
            continue
        if lower3 > 0.3 * rng3:
            continue
        # close3 not strongly extending the move (close near close2)
        if c3 <= c2:
            continue
        if (c3 - c2) > 0.5 * body2:
            continue
        out[i] = True
    return out


def cdl_stalledpattern(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Stalled Pattern (Deliberation) pattern.
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
        talib_out = talib.CDLSTALLEDPATTERN(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    mask = _cdl_stalledpattern_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_stalledpattern_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_STALLEDPATTERN",
) -> pl.DataFrame:
    """
    Add Stalled Pattern column to Polars DataFrame.
    """
    out = cdl_stalledpattern(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
