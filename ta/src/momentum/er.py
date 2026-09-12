# -*- coding: utf-8 -*-
"""Kaufman Efficiency Ratio (ER).

Measures how "efficient" price movement is over the last ``length``
bars:

    ER = |close[i] - close[i - length]| / sum(|close[j] - close[j-1]|)

ER is in [0, 1]: values near 1 mean a smooth directional trend,
values near 0 mean noisy chop.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates: a window touching a NaN close yields NaN ER;
- a fully flat window (numerator and denominator both zero) is
  ``0/0`` -> NaN (undefined), never a fabricated value.
"""
import numpy as np
import polars as pl

from numba import float64, int64, njit

from .._array_ops import _apply_offset_fillna


@njit((float64[:], int64), fastmath=False, cache=True)
def _er_numba(close: np.ndarray, length: int) -> np.ndarray:
    """Efficiency Ratio core (fastmath disabled: value-dependent
    ``denominator != 0.0`` guard).
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(length, n):
        change = np.abs(close[i] - close[i - length])
        volatility = 0.0
        bad = False
        for j in range(i - length + 1, i + 1):
            diff = close[j] - close[j - 1]
            if np.isnan(diff):
                bad = True
                break
            volatility += np.abs(diff)
        if bad:
            continue  # NaN window stays NaN (IEEE 754 propagation)
        if volatility != 0.0:
            out[i] = change / volatility
        # else: flat market -> 0/0 -> NaN (undefined)
    return out


def er_numpy(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numpy-based Efficiency Ratio calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Lookback window (>= 1). The first ``length`` bars are NaN.
    offset : int
        Shift of the output series.
    fillna : float, optional
        Replacement for warm-up/shifted-in NaNs.

    Returns
    -------
    np.ndarray
        ER values in [0, 1] (NaN for flat windows).

    Raises
    ------
    ValueError
        If ``length`` < 1.

    """
    if length < 1:
        raise ValueError('length must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.writeable:
        close = close.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _er_numba(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def er_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal Efficiency Ratio (accepts numpy array or Polars Series)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return er_numpy(close, length=length, offset=offset, fillna=fillna)


def er_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add Efficiency Ratio column to a Polars DataFrame.

    Default column name: ``ER_{length}``.
    """
    close = df[close_col].cast(pl.Float64).to_numpy()
    result = er_numpy(close, length=length, offset=offset, fillna=fillna)
    out_name = output_col or f'ER_{length}'
    return df.with_columns(pl.Series(out_name, result))
