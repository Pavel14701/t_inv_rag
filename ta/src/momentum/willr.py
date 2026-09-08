# -*- coding: utf-8 -*-
"""Williams %R (WILLR).

Momentum oscillator measuring where the close sits inside the
high-low range of the last ``length`` bars:

    WILLR = -100 * (HH - close) / (HH - LL)

Range is (-100, 0]; 0 means the close is at the highest high.
TA-Lib is used when available (semantics match exactly); otherwise a
Numba-backed native path runs silently.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- ``+/-Inf`` on input is converted to ``NaN``;
- NaN propagates through the rolling extremes and the division;
- a fully flat window (``HH == LL``) is 0/0-like: the result is NaN
  (undefined), never a fabricated value.
"""
import numpy as np
import polars as pl

from ..external import talib, talib_available
from .._array_ops import (
    _apply_offset_fillna,
    _rolling_max_numba,
    _rolling_min_numba,
)


def willr_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Numpy-based Williams %R calculation.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64), same length.
    length : int
        Lookback window (>= 1). The first ``length - 1`` bars are NaN.
    offset : int
        Shift of the output series.
    fillna : float, optional
        Replacement for warm-up/shifted-in NaNs.
    use_talib : bool
        Prefer TA-Lib when available.

    Returns
    -------
    np.ndarray
        WILLR values in (-100, 0].

    Raises
    ------
    ValueError
        If ``length`` < 1.

    """
    if length < 1:
        raise ValueError('length must be >= 1')
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Numba rolling kernels require writable contiguous buffers:
    # polars' zero-copy to_numpy() returns read-only arrays.
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.writeable:
        low = low.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)

    if use_talib and talib_available:
        result = talib.WILLR(high, low, close, timeperiod=length)
    else:
        highest_high = _rolling_max_numba(high, length)
        lowest_low = _rolling_min_numba(low, length)
        denom = highest_high - lowest_low
        with np.errstate(divide='ignore', invalid='ignore'):
            result = -100.0 * (highest_high - close) / denom
        # Flat window: range is zero, the ratio is undefined.
        result = np.where(denom == 0.0, np.nan, result)
    return _apply_offset_fillna(result, offset, fillna)


def willr_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Williams %R (accepts numpy array or Polars Series)."""
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return willr_numpy(
        high, low, close,
        length=length, offset=offset, fillna=fillna, use_talib=use_talib,
    )


def willr_polars(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add Williams %R column to a Polars DataFrame.

    Default column name: ``WILLR_{length}``.
    """
    high = df[high_col].cast(pl.Float64).to_numpy()
    low = df[low_col].cast(pl.Float64).to_numpy()
    close = df[close_col].cast(pl.Float64).to_numpy()
    result = willr_numpy(
        high, low, close,
        length=length, offset=offset, fillna=fillna, use_talib=use_talib,
    )
    out_name = output_col or f'WILLR_{length}'
    return df.with_columns(pl.Series(out_name, result))
