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
def _cdl_xsidegap3methods_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Numba-accelerated Upside/Downside Gap 3 Methods pattern.
    Returns boolean mask where pattern completes (True at the 3rd candle).
    Detects both Upside Gap 3 Methods (bullish continuation)
    and Downside Gap 3 Methods (bearish continuation).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(2, n):
        # Candle 1
        o1 = open_[i - 2]
        c1 = close[i - 2]
        # Candle 2
        o2 = open_[i - 1]
        c2 = close[i - 1]
        # Candle 3
        o3 = open_[i]
        c3 = close[i]
        # Bodies
        b1_low = min(o1, c1)
        b1_high = max(o1, c1)
        b2_low = min(o2, c2)
        b2_high = max(o2, c2)
        # --- Upside Gap 3 Methods (bullish continuation) ---
        # 1) Candle1 white, Candle2 white
        # 2) body gap up between 1 и 2
        # 3) Candle3 black, closing into the gap but not filling it
        bull = False
        if c1 > o1 and c2 > o2:
            if b2_low > b1_high:  # upside body gap
                if c3 < o3:  # third is black
                    if b1_high < c3 < b2_low:
                        bull = True
        # --- Downside Gap 3 Methods (bearish continuation) ---
        # 1) Candle1 black, Candle2 black
        # 2) body gap down between 1 и 2
        # 3) Candle3 white, closing into the gap but not filling it
        bear = False
        if c1 < o1 and c2 < o2:
            if b2_high < b1_low:  # downside body gap
                if c3 > o3:  # third is white
                    if b2_high < c3 < b1_low:
                        bear = True
        if bull or bear:
            out[i] = True
    return out


def cdl_xsidegap3methods(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Upside/Downside Gap 3 Methods pattern.
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
        talib_out = talib.CDLXSIDEGAP3METHODS(open_, high, low, close)
        talib_out = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(talib_out, offset, fillna)
    mask = _cdl_xsidegap3methods_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_xsidegap3methods_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_XSIDEGAP3METHODS",
) -> pl.DataFrame:
    """
    Add Upside/Downside Gap 3 Methods pattern column to Polars DataFrame.
    """
    out = cdl_xsidegap3methods(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
