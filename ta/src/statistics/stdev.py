"""Rolling standard deviation (STDEV) for financial time series.

This module provides Numba-accelerated and TA-Lib implementations of
rolling standard deviation, with unified interface for numpy arrays
and Polars Series/DataFrames.

Functions:
    stdev_numba: Numba-accelerated rolling standard deviation.
    stdev_talib: TA-Lib-based rolling standard deviation (ddof=0).
    stdev_ind: Universal rolling standard deviation (numpy or Polars Series).
    stdev_polars: Add rolling standard deviation column to Polars DataFrame.
    stdev_polars_multi: Add rolling standard deviation columns
    for multiple columns.

The core algorithm is implemented in Numba for high performance.
"""

from typing import Literal

import numpy as np
import polars as pl
from numba import njit

from ..external import talib, talib_available
from .._array_ops import _apply_offset_fillna


@njit('float64[:](float64[:], int64, int64)', fastmath=True, cache=True)
def _stdev_numba_core_online(
    close: np.ndarray,
    length: int,
    ddof: int
) -> np.ndarray:
    """Online (one-pass) rolling standard deviation.

    Uses running sums and sums of squares for O(1) update per element.
    Fast but may have slight numerical inaccuracies for large windows.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(length):
        val = close[i]
        sum_x += val
        sum_x2 += val * val
    mean = sum_x / length
    variance = (
        (sum_x2 - 2 * mean * sum_x + length * mean * mean)
        / (length - ddof)
    )
    out[length - 1] = np.sqrt(variance) if variance >= 0 else np.nan
    for i in range(length, n):
        new_val = close[i]
        old_val = close[i - length]
        sum_x += new_val - old_val
        sum_x2 += new_val * new_val - old_val * old_val
        mean = sum_x / length
        variance = (
            (sum_x2 - 2 * mean * sum_x + length * mean * mean)
            / (length - ddof)
        )
        out[i] = np.sqrt(variance) if variance >= 0 else np.nan
    return out


@njit('float64[:](float64[:], int64, int64)', fastmath=True, cache=True)
def _stdev_numba_core_twopass(
    close: np.ndarray,
    length: int,
    ddof: int
) -> np.ndarray:
    """Two-pass rolling standard deviation.

    Computes mean first, then variance. Slower but more numerically stable.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        # Compute mean
        sum_x = 0.0
        for j in range(i - length + 1, i + 1):
            sum_x += close[j]
        mean = sum_x / length
        # Compute variance
        sum_sq = 0.0
        for j in range(i - length + 1, i + 1):
            diff = close[j] - mean
            sum_sq += diff * diff
        variance = sum_sq / (length - ddof)
        out[i] = np.sqrt(variance) if variance >= 0 else np.nan
    return out


def _stdev_numba_core(
    close: np.ndarray,
    length: int,
    ddof: int,
    algorithm: Literal['online', 'two_pass'] = 'online',
) -> np.ndarray:
    """Dispatch to the appropriate Numba core function."""
    if algorithm == 'online':
        return _stdev_numba_core_online(close, length, ddof)
    else:
        return _stdev_numba_core_twopass(close, length, ddof)


def stdev_numba(
    close: np.ndarray,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    algorithm: Literal['online', 'two_pass'] = 'online',
) -> np.ndarray:
    """Numba-accelerated rolling standard deviation.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size.
    ddof : int, default 1
        Delta Degrees of Freedom (1 for sample std, 0 for population).
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.
    algorithm : {'online', 'two_pass'}, default 'online'
        - 'online': one-pass algorithm (fast, may have small errors).
        - 'two_pass': two-pass algorithm (slower, more accurate).

    Returns
    -------
    np.ndarray
        Float64 array of rolling standard deviations, shifted and NaN-filled.

    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.writeable:
        close = close.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _stdev_numba_core(close, length, ddof, algorithm)
    return _apply_offset_fillna(result, offset, fillna)


def stdev_talib(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """TA-Lib-based rolling standard deviation (ddof=0)."""
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = talib.STDDEV(close, timeperiod=length)
    return _apply_offset_fillna(result, offset, fillna)


def stdev_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    algorithm: Literal['online', 'two_pass'] = 'online',
) -> np.ndarray:
    """Universal rolling standard deviation (numpy array or Polars Series).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        1D array or Polars Series of close prices.
    length : int, default 30
        Window size.
    ddof : int, default 1
        Delta Degrees of Freedom (1 for sample, 0 for population).
    offset : int, default 0
        Shift applied to the output array.
    fillna : float or None, default None
        Value to fill NaN positions after offset.
    use_talib : bool, default True
        If True and TA-Lib is available, use TA-Lib (ddof=0).
        Otherwise, use Numba.
    algorithm : {'online', 'two_pass'}, default 'online'
        Only used when use_talib=False. See stdev_numba.

    Returns
    -------
    np.ndarray
        Float64 array of rolling standard deviations.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return stdev_talib(close, length, offset, fillna)
    else:
        return stdev_numba(close, length, ddof, offset, fillna, algorithm)


def stdev_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    algorithm: Literal['online', 'two_pass'] = 'online',
    output_col: str | None = None,
) -> pl.Series:
    """Add rolling standard deviation column to a Polars DataFrame."""
    close = df[close_col].to_numpy()
    result = stdev_ind(
        close,
        length=length,
        ddof=ddof,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        algorithm=algorithm,
    )
    out_name = output_col or f'STDEV_{length}'
    return pl.Series(out_name, result)


def stdev_polars_multi(
    df: pl.DataFrame,
    columns: list[str],
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = '_stdev',
    use_talib: bool = False,
    algorithm: Literal['online', 'two_pass'] = 'online',
) -> pl.DataFrame:
    """Add rolling standard deviation columns for multiple columns."""
    for col in columns:
        arr = df[col].to_numpy()
        result = stdev_ind(
            arr,
            length=length,
            ddof=ddof,
            offset=offset,
            fillna=fillna,
            use_talib=use_talib,
            algorithm=algorithm,
        )
        df = df.with_columns(pl.Series(f'{col}{suffix}', result))
    return df
