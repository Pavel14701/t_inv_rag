# -*- coding: utf-8 -*-
"""Rolling Mean Absolute Deviation (MAD) for financial time series.

Mean Absolute Deviation measures the average absolute deviation of prices
from their mean over a rolling window.  It is a robust measure of dispersion
that is less sensitive to outliers than the standard deviation.

This module provides Numba-accelerated computation of rolling MAD.

Functions:
    mad_numba: Numba-accelerated rolling MAD.
    mad_ind: Universal rolling MAD (numpy or Polars Series).
    mad_polars: Add MAD column to Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import _apply_offset_fillna


@jit(nopython=True, fastmath=False, cache=True)
def _mad_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-compiled core for rolling Mean Absolute Deviation.

    For each window, the mean is recomputed from scratch (two-pass) and
    then the mean absolute deviation is calculated.  This implementation
    uses O(n * length) time, which is required for strict IEEE 754
    compliance: incremental running sums accumulate floating-point drift
    and a single NaN/inf would permanently poison every later value,
    whereas the two-pass form propagates NaN/inf only while the affected
    value is inside the window and stays exact on large-magnitude inputs.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 2).

    Returns
    -------
    np.ndarray
        Float64 array of rolling MAD values, with first `length-1` elements
        set to NaN.  NaN/inf inputs propagate to windows containing them.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out

    for i in range(length - 1, n):
        # Fresh summation per window: no running-sum drift, and a NaN/inf
        # affects only the windows that contain it.
        s = 0.0
        for j in range(i - length + 1, i + 1):
            s += close[j]
        mean = s / length

        mad = 0.0
        for j in range(i - length + 1, i + 1):
            mad += abs(close[j] - mean)
        out[i] = mad / length

    return out


def mad_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numba-accelerated rolling Mean Absolute Deviation.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size (must be >= 2).
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling MAD values, shifted and NaN-filled
        according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> mad_numba(prices, length=3)
    array([       nan,        nan, 0.66666667, 0.66666667, 0.66666667, 0.66666667])

    """  # noqa: E501
    close = np.asarray(close, dtype=np.float64)
    if length < 2:
        raise ValueError("length must be >= 2")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()

    result = _mad_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def mad_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal rolling MAD (numpy array or Polars Series).

    This is a wrapper around :func:`mad_numba` that automatically converts
    Polars Series to numpy arrays before processing.  All other parameters
    behave exactly as in :func:`mad_numba`.

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
        Float64 array of rolling MAD values.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> mad_ind(s, length=3)
    array([       nan,        nan, 0.66666667, 0.66666667, 0.66666667, 0.66666667])

    """  # noqa: E501
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return mad_numba(close, length, offset, fillna)


def mad_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """Add a rolling Mean Absolute Deviation column to a Polars DataFrame.

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
        Name of the output column. If None, uses f'MAD_{length}'.

    Returns
    -------
    pl.Series
        A new Polars Series containing the rolling MAD.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    >>> mad_polars(df, length=3)
    shape: (6,)
    Series: 'MAD_3' [f64]
    [
        null
        null
        0.666667
        0.666667
        0.666667
        0.666667
    ]

    """
    close = df[close_col].to_numpy()
    result = mad_ind(close, length, offset, fillna)
    out_name = output_col or f"MAD_{length}"
    return pl.Series(out_name, result)
