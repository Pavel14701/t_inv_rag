# -*- coding: utf-8 -*-
"""Fisher Transform.

Transforms prices into a Gaussian-like distribution via a recursive
normalisation of the position of hlc3 inside the high-low range:

    raw[i]   = 0.66 * ((hlc3 - LL) / (HH - LL) - 0.5) + 0.67 * raw[i-1]
    value[i] = 0.5 * (raw[i] + value[i-1])
    fisher[i] = 0.5 * ln((1 + value) / (1 - value))

The second output is the signal line (Fisher shifted by 1 bar).

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates through the whole recursion (a single NaN window
  poisons every later bar — this matches pandas_ta behaviour);
- a flat window (``HH == LL``) yields NaN (0/0 undefined);
- ``|value| >= 1`` makes the log argument non-positive: log(0) -> -Inf,
  log(negative) -> NaN (documented, never raised).
"""
import numpy as np
import polars as pl
from numba import float64, int64, njit

from .._array_ops import (
    _apply_offset_fillna,
    _rolling_max_numba,
    _rolling_min_numba,
)


@njit(
    (float64[:], float64[:], float64[:], int64),
    fastmath=False,
    cache=True,
)
def _fisher_numba(
    hlc3: np.ndarray,
    highest: np.ndarray,
    lowest: np.ndarray,
    length: int,
) -> np.ndarray:
    """Recursive Fisher transform core.

    Seeds raw/value with 0 at the first fully valid bar; everything
    before that is NaN.
    """
    n = len(hlc3)
    fisher = np.full(n, np.nan, dtype=np.float64)
    raw_prev = 0.0
    value_prev = 0.0
    started = False
    for i in range(n):
        if i < length - 1:
            continue
        denom = highest[i] - lowest[i]
        if np.isnan(denom) or denom == 0.0:
            if started:
                # NaN propagates through the recursion
                raw_prev = np.nan
                value_prev = np.nan
            continue
        raw = 0.66 * ((hlc3[i] - lowest[i]) / denom - 0.5) + 0.67 * raw_prev
        value = 0.5 * (raw + value_prev)
        arg = (1.0 + value) / (1.0 - value)
        fisher[i] = 0.5 * np.log(arg)
        raw_prev = raw
        value_prev = value
        started = True
    return fisher


def fisher_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 9,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Fisher Transform and its signal line using NumPy.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64), same length. The normalised value is
        the position of hlc3 = (high + low + close) / 3 inside the
        high/low range.
    length : int
        Rolling window for the high/low range (>= 1). The first
        ``length - 1`` bars are NaN.
    offset, fillna : as usual.

    Returns
    -------
    tuple of np.ndarray
        (fisher, signal) where signal = fisher shifted by 1 bar.

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
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.writeable:
        low = low.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    hlc3 = (high + low + close) / 3.0
    highest = _rolling_max_numba(high, length)
    lowest = _rolling_min_numba(low, length)
    fisher = _fisher_numba(hlc3, highest, lowest, length)
    # Signal line: Fisher shifted by one bar.
    signal = np.full(len(fisher), np.nan, dtype=np.float64)
    if len(fisher) > 1:
        signal[1:] = fisher[:-1]
    fisher = _apply_offset_fillna(fisher, offset, fillna)
    signal = _apply_offset_fillna(signal, offset, fillna)
    return fisher, signal


def fisher_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 9,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal Fisher Transform (accepts numpy array or Polars Series)."""
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return fisher_numpy(
        high, low, close, length=length, offset=offset, fillna=fillna,
    )


def fisher_polars(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    length: int = 9,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = '',
) -> pl.DataFrame:
    """Add Fisher Transform columns to a Polars DataFrame.

    Added columns: ``FISHERT{suffix}`` and ``FISHERTs{suffix}`` where
    suffix defaults to ``_{length}``.
    """
    high = df[high_col].cast(pl.Float64).to_numpy()
    low = df[low_col].cast(pl.Float64).to_numpy()
    close = df[close_col].cast(pl.Float64).to_numpy()
    fisher, signal = fisher_numpy(
        high, low, close, length=length, offset=offset, fillna=fillna,
    )
    if not suffix:
        suffix = f'_{length}'
    return df.with_columns([
        pl.Series(f'FISHERT{suffix}', fisher),
        pl.Series(f'FISHERTs{suffix}', signal),
    ])
