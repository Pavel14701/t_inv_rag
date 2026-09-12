"""Rolling variance (VAR) for financial time series.

This module provides Numba-accelerated and TA-Lib implementations of
rolling variance, with unified interface for numpy arrays and Polars
Series/DataFrames.

Variance is the square of standard deviation.  It measures the dispersion
of prices around their mean over a rolling window.

Functions:
    variance_numba: Numba-accelerated rolling variance.
    variance_talib: TA-Lib-based rolling variance (ddof=0).
    variance_ind: Universal rolling variance (numpy or Polars Series).
    variance_polars: Add rolling variance column to Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import _apply_offset_fillna
from ..external import talib, talib_available


@jit(nopython=True, fastmath=False, cache=True)
def _variance_numba_core(
    close: np.ndarray,
    length: int,
    ddof: int
) -> np.ndarray:
    """Numba-compiled core for rolling variance.

    For each window the mean is recomputed from scratch and the variance is
    taken from the summed squared deviations (two-pass).  This uses
    O(n * length) time, which is required for strict IEEE 754 compliance:
    incremental running sums of squares accumulate floating-point drift and
    a single NaN/inf would permanently poison every later value, whereas
    the two-pass form propagates NaN/inf only while the affected value is
    inside the window and stays exact on large-magnitude inputs.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 1).
    ddof : int
        Delta Degrees of Freedom (must satisfy 0 <= ddof < length).

    Returns
    -------
    np.ndarray
        Float64 array of rolling variances, with first `length-1` elements
        set to NaN.  NaN/inf inputs propagate to windows containing them.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    denom = length - ddof
    for i in range(length - 1, n):
        # Fresh summation per window: no running-sum drift, and a NaN/inf
        # affects only the windows that contain it.
        s = 0.0
        for j in range(i - length + 1, i + 1):
            s += close[j]
        mean = s / length

        ss = 0.0
        for j in range(i - length + 1, i + 1):
            d = close[j] - mean
            ss += d * d
        # Sum of squared deviations is non-negative by construction.
        out[i] = ss / denom
    return out


def variance_numba(
    close: np.ndarray,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numba-accelerated rolling variance.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size.
    ddof : int, default 1
        Delta Degrees of Freedom (1 for sample variance, 0 for population).
        Must satisfy 0 <= ddof < length.
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling variances, shifted and NaN-filled
        according to `offset` and `fillna`.

    Examples
    --------
    >>> close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> variance_numba(close, length=3, ddof=1)
    array([       nan,        nan, 0.66666667, 0.66666667, 0.66666667])

    """
    close = np.asarray(close, dtype=np.float64)
    if length < 1:
        raise ValueError('length must be >= 1')
    if ddof < 0 or ddof >= length:
        raise ValueError('ddof must satisfy 0 <= ddof < length')
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()

    result = _variance_numba_core(close, length, ddof)
    return _apply_offset_fillna(result, offset, fillna)


def variance_talib(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """TA-Lib-based rolling variance.

    Note: TA-Lib uses ddof=0 (population variance).

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size.
    offset : int, default 0
        Shift applied to the output array.
    fillna : float or None, default None
        Value to fill NaN positions after offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling variances.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.

    """
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    close = np.asarray(close, dtype=np.float64)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()

    result = talib.VAR(close, timeperiod=length)
    return _apply_offset_fillna(result, offset, fillna)


def variance_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal rolling variance (numpy array or Polars Series).

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
        Otherwise, use Numba implementation.

    Returns
    -------
    np.ndarray
        Float64 array of rolling variances.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> variance_ind(s, length=3, ddof=1)
    array([       nan,        nan, 0.66666667, 0.66666667, 0.66666667])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    if use_talib and talib_available:
        return variance_talib(close, length, offset, fillna)
    else:
        return variance_numba(close, length, ddof, offset, fillna)


def variance_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.Series:
    """Add a rolling variance column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 30
        Window size.
    ddof : int, default 1
        Delta Degrees of Freedom.
    offset : int, default 0
        Shift applied to the output column.
    fillna : float or None, default None
        Value to fill NaN positions after offset.
    use_talib : bool, default True
        Whether to use TA-Lib (if available).
    output_col : str or None, default None
        Name of the output column. If None, uses f'VAR_{length}'.

    Returns
    -------
    pl.Series
        A new Polars Series containing the rolling variance.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> variance_polars(df, length=3, ddof=1)
    shape: (5,)
    Series: 'VAR_3' [f64]
    [
        null
        null
        0.666667
        0.666667
        0.666667
    ]

    """
    close = df[close_col].to_numpy()
    result = variance_ind(
        close,
        length=length,
        ddof=ddof,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )
    out_name = output_col or f'VAR_{length}'
    return pl.Series(out_name, result)
