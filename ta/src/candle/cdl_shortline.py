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
def _cdl_shortline_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Numba‑accelerated Short Line Candle pattern (TA‑Lib semantics).

    A candle is a Short Line when its real body, upper and lower shadows are
    each smaller than 0.3x the corresponding average over the previous 5
    candles. Returns boolean mask where pattern occurs.
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    period = 5
    factor = 0.3
    for i in range(period, n):
        body_sum = 0.0
        upper_sum = 0.0
        lower_sum = 0.0
        for k in range(i - period, i):
            b = abs(close[k] - open_[k])
            u = high[k] - max(open_[k], close[k])
            lo = min(open_[k], close[k]) - low[k]
            body_sum += b
            upper_sum += u
            lower_sum += lo
        body = abs(close[i] - open_[i])
        upper = high[i] - max(open_[i], close[i])
        lower = min(open_[i], close[i]) - low[i]
        if (
            body < factor * (body_sum / period)
            and upper < factor * (upper_sum / period)
            and lower < factor * (lower_sum / period)
        ):
            out[i] = True
    return out


def cdl_shortline(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low_: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Short Line Candle pattern.
    Returns numpy array of float64: 1.0 where pattern occurs, else 0.0.
    """
    if isinstance(open_, pl.Series): 
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series): 
        high = high.to_numpy()
    if isinstance(low_, pl.Series): 
        low_ = low_.to_numpy()
    if isinstance(close, pl.Series): 
        close = close.to_numpy()
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low_ = np.asarray(low_, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not open_.flags.writeable:
        open_ = open_.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not high.flags.writeable:
        high = high.copy()
    if not low_.flags.c_contiguous:
        low_ = np.ascontiguousarray(low_)
    if not low_.flags.writeable:
        low_ = low_.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    if use_talib and talib_available:
        talib_out = talib.CDLSHORTLINE(open_, high, low_, close)
        talib_out = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(talib_out, offset, fillna)
    mask = _cdl_shortline_nb(open_, high, low_, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_shortline_polars(
    df: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = 'CDL_SHORTLINE',
) -> pl.DataFrame:
    """Add Short Line Candle column to Polars DataFrame."""
    out = cdl_shortline(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
