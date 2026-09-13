# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from numba import float64, njit

from .._array_ops import _apply_offset_fillna
from ..external import talib, talib_available


@njit((float64[:], float64[:], float64[:], float64[:]), cache=True)
def _cdl_risefall3methods_nb(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Numba-accelerated Rise/Fall 3 Methods pattern.
    Returns boolean mask where pattern completes (True at the 5th candle).
    Detects both Rising Three Methods (bullish) and Falling Three Methods
        (bearish).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.bool_)
    for i in range(4, n):
        # Candle 1
        o1 = open_[i - 4]
        c1 = close[i - 4]
        h1 = high[i - 4]
        l1 = low[i - 4]
        rng1 = h1 - l1
        body1 = abs(c1 - o1)
        if rng1 <= 0.0 or body1 < 0.6 * rng1:
            continue  # must be long candle
        # Candles 2-4
        small_ok_bull = True
        small_ok_bear = True
        for k in range(3, 0, -1):  # i-3, i-2, i-1
            o = open_[i - k]
            c = close[i - k]
            h = high[i - k]
            low_ = low[i - k]
            rng = h - low_
            body = abs(c - o)
            if rng <= 0.0:
                small_ok_bull = False
                small_ok_bear = False
                break
            # small bodies relative to first
            if body > 0.4 * body1:
                small_ok_bull = False
                small_ok_bear = False
                break
            # must stay within range of candle 1
            if h > h1 or low_ < l1:
                small_ok_bull = False
                small_ok_bear = False
                break
            # For rising three methods: small counter-trend (bearish) candles
            if not (c < o):
                small_ok_bull = False
            # For falling three methods: small counter-trend (bullish) candles
            if not (c > o):
                small_ok_bear = False
        # Candle 5
        o5 = open_[i]
        c5 = close[i]
        h5 = high[i]
        l5 = low[i]
        rng5 = h5 - l5
        body5 = abs(c5 - o5)
        if rng5 <= 0.0 or body5 < 0.6 * rng5:
            continue  # must be long candle
        bullish_pattern = False
        bearish_pattern = False
        # Rising Three Methods (bullish continuation)
        if c1 > o1 and small_ok_bull and c5 > o5 and c5 > c1:
            bullish_pattern = True
        # Falling Three Methods (bearish continuation)
        if c1 < o1 and small_ok_bear and c5 < o5 and c5 < c1:
            bearish_pattern = True
        if bullish_pattern or bearish_pattern:
            out[i] = True
    return out


def cdl_risefall3methods(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Rise/Fall 3 Methods pattern.
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
        talib_out = talib.CDLRISEFALL3METHODS(open_, high, low, close)
        result = (talib_out != 0).astype(np.float64)
        return _apply_offset_fillna(result, offset, fillna)
    mask = _cdl_risefall3methods_nb(open_, high, low, close)
    out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_risefall3methods_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_RISEFALL3METHODS",
) -> pl.DataFrame:
    """Add Rise/Fall 3 Methods column to Polars DataFrame."""
    out = cdl_risefall3methods(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
