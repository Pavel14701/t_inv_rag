# -*- coding: utf-8 -*-
"""Psychological Line (PSL).

Percentage of strictly up bars over the last ``length`` bars:

    PSL = 100 * count(close[j] - close[j - drift] > 0) / length

PSL is in [0, 100]; values near 100 signal over-optimism, near 0
over-pessimism (a contrarian oscillator).

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- a window touching a NaN close is undefined and yields NaN;
- zero-change bars count as neither up nor down (neutral).
"""
import numpy as np
import polars as pl

from numba import float64, int64, njit

from .._array_ops import _apply_offset_fillna


@njit((float64[:], int64, int64), fastmath=False, cache=True)
def _psl_numba(close: np.ndarray, length: int, drift: int) -> np.ndarray:
    """Psychological Line core (fastmath disabled: value-dependent
    ``NaN``/``> 0`` comparisons).
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    first = length + drift - 1
    for i in range(first, n):
        up = 0.0
        bad = False
        for j in range(i - length + 1, i + 1):
            diff = close[j] - close[j - drift]
            if np.isnan(diff):
                bad = True
                break
            if diff > 0.0:
                up += 1.0
        if bad:
            continue  # NaN window stays NaN (IEEE 754 propagation)
        out[i] = 100.0 * up / length
    return out


def psl_numpy(
    close: np.ndarray,
    length: int = 12,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numpy-based Psychological Line calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length (>= 1). Warm-up: first ``length + drift - 1``
        bars are NaN.
    drift : int
        Lookback for the per-bar change (>= 1).
    offset : int
        Shift of the output series.
    fillna : float, optional
        Replacement for warm-up/shifted-in NaNs.

    Returns
    -------
    np.ndarray
        PSL values in [0, 100].

    Raises
    ------
    ValueError
        If ``length`` < 1 or ``drift`` < 1.

    """
    if length < 1:
        raise ValueError('length must be >= 1')
    if drift < 1:
        raise ValueError('drift must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.writeable:
        close = close.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _psl_numba(close, length, drift)
    return _apply_offset_fillna(result, offset, fillna)


def psl_ind(
    close: np.ndarray | pl.Series,
    length: int = 12,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal Psychological Line (accepts numpy array or Polars Series)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return psl_numpy(
        close, length=length, drift=drift, offset=offset, fillna=fillna,
    )


def psl_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 12,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add Psychological Line column to a Polars DataFrame.

    Default column name: ``PSL_{length}``.
    """
    close = df[close_col].cast(pl.Float64).to_numpy()
    result = psl_numpy(
        close, length=length, drift=drift, offset=offset, fillna=fillna,
    )
    out_name = output_col or f'PSL_{length}'
    return df.with_columns(pl.Series(out_name, result))
