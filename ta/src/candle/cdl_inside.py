# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from numba import float64, njit

from .._array_ops import _apply_offset_fillna


@njit((float64[:], float64[:], float64[:], float64[:]), cache=True)
def _cdl_inside_nb(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Numba-accelerated Inside pattern.
    Returns boolean mask where pattern completes (True at the second candle).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        # Candle 1 (i-1)
        h1 = high[i - 1]
        l1 = low[i - 1]
        # Candle 2 (i)
        h2 = high[i]
        l2 = low[i]
        # Inside bar: full containment
        if h2 < h1 and l2 > l1:
            out[i] = True
    return out


def cdl_inside(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,  # ignored
) -> np.ndarray:
    """Universal Inside pattern.
    Returns numpy array of float64: 1.0 where pattern occurs, else 0.0.
    """
    # Polars -> numpy
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    # Ensure float64 + contiguous
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
    # Numba branch
    mask = _cdl_inside_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_inside_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_INSIDE",
) -> pl.DataFrame:
    """Add Inside pattern column to Polars DataFrame."""
    out = cdl_inside(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
