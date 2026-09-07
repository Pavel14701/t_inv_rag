# -*- coding: utf-8 -*-
"""Rolling excess kurtosis (KURT) for financial time series.

Kurtosis measures the "tailedness" of the distribution of returns or prices
over a rolling window.  Positive excess kurtosis (leptokurtic) indicates
heavy tails and more extreme outliers, while negative excess kurtosis
(platykurtic) indicates light tails.

This module uses the standard bias-corrected excess kurtosis estimator
(unbiased for normal distributions), computed per window with a two-pass
central-moments algorithm for strict IEEE 754 compliance.

Functions:
    kurtosis_numba: Numba-accelerated rolling kurtosis.
    kurtosis_ind: Universal rolling kurtosis (numpy or Polars Series).
    kurtosis_polars: Add kurtosis column to Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl
from numba import jit

from .._array_ops import _apply_offset_fillna


@jit(nopython=True, fastmath=False, cache=True)
def _kurtosis_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-compiled core for rolling excess kurtosis.

    Computes, per window, the sample mean and the central moments M2/M4
    with a two-pass algorithm and returns the standard bias-corrected
    (unbiased for normal distributions) excess kurtosis:

        G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)

    where ``g2 = M4 / M2**2 - 3`` is the population excess kurtosis.
    This matches ``scipy.stats.kurtosis(..., bias=False)``.

    Follows IEEE 754 strictly (``fastmath=False``): NaN/inf inputs
    propagate to the windows that contain them, and later windows
    recover once the non-finite value leaves the window.  A constant
    window (M2 == 0) yields NaN.

    The O(n * length) two-pass form is intentional: incremental running
    power sums both accumulate floating-point drift (catastrophic
    cancellation on large-magnitude prices) and would be permanently
    poisoned by a single NaN.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 4; below that the estimator is undefined).

    Returns
    -------
    np.ndarray
        Float64 array of rolling excess kurtosis, with first `length-1`
        elements set to NaN.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length or length < 4:
        return out

    for i in range(length - 1, n):
        # Fresh per-window pass: no running-sum drift, NaN/inf affects
        # only the windows that contain it.
        mean = 0.0
        for j in range(i - length + 1, i + 1):
            mean += close[j]
        mean /= length

        m2 = 0.0
        m4 = 0.0
        for j in range(i - length + 1, i + 1):
            d = close[j] - mean
            d2 = d * d
            m2 += d2
            m4 += d2 * d2

        if not np.isfinite(m2) or m2 <= 0.0:
            # m2 == 0 -> constant window; non-finite -> NaN/inf input
            continue

        g2 = (length * m4) / (m2 * m2) - 3.0
        out[i] = (
            (length - 1)
            * ((length + 1) * g2 + 6.0)
            / ((length - 2) * (length - 3))
        )

    return out


def kurtosis_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numba-accelerated rolling excess kurtosis.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size (must be >= 4 for finite value).
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling excess kurtosis, shifted and NaN-filled
        according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> kurtosis_numba(prices, length=4)
    array([ nan,  nan,  nan, -1.2, -1.2, -1.2])

    """
    close = np.asarray(close, dtype=np.float64)
    if length < 4:
        raise ValueError('length must be >= 4')
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()

    result = _kurtosis_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def kurtosis_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal rolling excess kurtosis (numpy array or Polars Series).

    This is a wrapper around :func:`kurtosis_numba` that automatically converts
    Polars Series to numpy arrays before processing.  All other parameters
    behave exactly as in :func:`kurtosis_numba`.

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
        Float64 array of rolling excess kurtosis.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> kurtosis_ind(s, length=4)
    array([ nan,  nan,  nan, -1.2, -1.2, -1.2])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return kurtosis_numba(close, length, offset, fillna)


def kurtosis_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add a rolling excess kurtosis column to a Polars DataFrame.

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
        Name of the output column. If None, uses f'KURT_{length}'.

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the rolling excess kurtosis column appended.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    >>> kurtosis_polars(df, length=4)
    shape: (6, 2)
    ┌───────┬──────────┐
    │ close ┆ KURT_4   │
    │ ---   ┆ ---      │
    │ f64   ┆ f64      │
    ╞═══════╪══════════╡
    │ 1.0   ┆ NaN      │
    │ 2.0   ┆ NaN      │
    │ 3.0   ┆ NaN      │
    │ 4.0   ┆ -1.2     │
    │ 5.0   ┆ -1.2     │
    │ 6.0   ┆ -1.2     │
    └───────┴──────────┘

    """
    close = df[close_col].to_numpy()
    result = kurtosis_ind(close, length, offset, fillna)
    out_name = output_col or f'KURT_{length}'
    return df.with_columns(pl.Series(out_name, result))
