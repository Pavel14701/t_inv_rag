# -*- coding: utf-8 -*-
"""Rolling skewness (SKEW) for financial time series.

Skewness measures the asymmetry of the distribution of returns or prices
over a rolling window.  Positive skewness indicates a longer right tail
(upward outliers), while negative skewness indicates a longer left tail
(downward outliers).

This module provides Numba-accelerated computation of rolling skewness
using the Fisher-Pearson coefficient (bias-corrected).  The implementation
maintains running sums of powers for O(1) update per element.

Functions:
    skew_numba: Numba-accelerated rolling skewness.
    skew_ind: Universal rolling skewness (numpy or Polars Series).
    skew_polars: Add skewness column to Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl
from numba import njit

from .._array_ops import _apply_offset_fillna


@njit(fastmath=True, cache=True)
def _skew_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-compiled core for rolling skewness.

    Uses running sums of powers (sum x, sum x², sum x³) for O(1) update per
    element.  The skewness is computed using the Fisher-Pearson coefficient
    (bias-corrected for sample data).

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 3).

    Returns
    -------
    np.ndarray
        Float64 array of rolling skewness, with first `length-1` elements
        set to NaN.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial sums for the first window
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for i in range(length):
        x = close[i]
        sum1 += x
        sum2 += x * x
        sum3 += x * x * x

    # Helper to compute skewness from sums (reusable)
    def compute_skew(s1, s2, s3, L):  # noqa: N803
        if L < 3:
            return np.nan
        mean = s1 / L
        # Central moments (not normalised)
        m2 = s2 - L * mean * mean
        m3 = s3 - 3.0 * mean * s2 + 2.0 * L * mean * mean * mean
        if m2 <= 0.0:
            return np.nan
        std = np.sqrt(m2 / (L - 1))  # sample standard deviation
        # Fisher-Pearson skewness (bias-corrected)
        return (L * m3) / ((L - 1) * (L - 2) * (std ** 3))

    out[length - 1] = compute_skew(sum1, sum2, sum3, length)
    # Sliding update
    for i in range(length, n):
        add = close[i]
        rem = close[i - length]
        sum1 += add - rem
        sum2 += add * add - rem * rem
        sum3 += add * add * add - rem * rem * rem
        out[i] = compute_skew(sum1, sum2, sum3, length)
    return out


def skew_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numba-accelerated rolling skewness.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size (must be >= 3 for a meaningful skewness).
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling skewness, shifted and NaN-filled
        according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> skew_numba(prices, length=4)
    array([nan, nan, nan, 0. , 0. , 0.])

    """
    close = np.asarray(close, dtype=np.float64)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()

    result = _skew_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def skew_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal rolling skewness (numpy array or Polars Series).

    This is a wrapper around :func:`skew_numba` that automatically converts
    Polars Series to numpy arrays before processing.  All other parameters
    behave exactly as in :func:`skew_numba`.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        1D array or Polars Series of close prices.
    length : int, default 30
        Window size.
    offset : int, default 0
        Shift applied to the output.
    fillna : float or None, default None
        Value to fill NaN positions after offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling skewness.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> skew_ind(s, length=4)
    array([nan, nan, nan, 0. , 0. , 0. ])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return skew_numba(close, length, offset, fillna)


def skew_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """Add a rolling skewness column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 30
        Window size.
    offset : int, default 0
        Shift applied to the output column.
    fillna : float or None, default None
        Value to fill NaN positions after offset.
    output_col : str or None, default None
        Name of the output column. If None, uses f'SKEW_{length}'.

    Returns
    -------
    pl.Series
        A new Polars Series containing the rolling skewness.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    >>> skew_polars(df, length=4)
    shape: (6,)
    Series: 'SKEW_4' [f64]
    [
        null
        null
        null
        0.0
        0.0
        0.0
    ]

    """
    close = df[close_col].to_numpy()
    result = skew_ind(close, length, offset, fillna)
    out_name = output_col or f'SKEW_{length}'
    return pl.Series(out_name, result)
