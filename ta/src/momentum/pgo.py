# -*- coding: utf-8 -*-
"""Pretty Good Oscillator (PGO).

    PGO = (close - EMA(close, length)) / (HH - LL)

where HH/LL are the rolling extremes of high/low over ``length`` bars.
Values above +3 / below -3 signal breakouts relative to recent
volatility.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates through the EMA, the rolling extremes and the ratio;
- a fully flat window (``HH == LL``) is undefined: the result is NaN
  (explicit rule, wins over ``x/0`` -> +/-Inf).
"""
import numpy as np
import polars as pl

from ..overlap.ema import ema_ind
from .._array_ops import (
    _apply_offset_fillna,
    _rolling_max_numba,
    _rolling_min_numba,
)


def pgo_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Numpy-based Pretty Good Oscillator calculation.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64), same length.
    length : int
        EMA and rolling-range window (>= 1). Warm-up: first
        ``length - 1`` bars are NaN.
    offset, fillna : as usual.
    use_talib : bool
        Prefer TA-Lib for the inner EMA when available.

    Returns
    -------
    np.ndarray
        PGO values (NaN on flat windows).

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
    if high.size == 0:
        return np.array([])
    # Numba rolling kernels require writable contiguous buffers.
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.writeable:
        low = low.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    ema = ema_ind(
        close, length=length, use_talib=use_talib, nan_policy='ignore'
    )
    highest = _rolling_max_numba(high, length)
    lowest = _rolling_min_numba(low, length)
    denom = highest - lowest
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (close - ema) / denom
    # Flat window: range is zero, the ratio is undefined.
    result = np.where(denom == 0.0, np.nan, result)
    return _apply_offset_fillna(result, offset, fillna)


def pgo_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal PGO (accepts numpy array or Polars Series)."""
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return pgo_numpy(
        high, low, close,
        length=length, offset=offset, fillna=fillna, use_talib=use_talib,
    )


def pgo_polars(
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
    """Add Pretty Good Oscillator column to a Polars DataFrame.

    Default column name: ``PGO_{length}``.
    """
    high = df[high_col].cast(pl.Float64).to_numpy()
    low = df[low_col].cast(pl.Float64).to_numpy()
    close = df[close_col].cast(pl.Float64).to_numpy()
    result = pgo_numpy(
        high, low, close,
        length=length, offset=offset, fillna=fillna, use_talib=use_talib,
    )
    out_name = output_col or f'PGO_{length}'
    return df.with_columns(pl.Series(out_name, result))
