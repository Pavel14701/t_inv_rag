# -*- coding: utf-8 -*-
"""Rolling excess kurtosis (KURT) for financial time series.

Kurtosis measures the "tailedness" of the distribution of returns or prices
over a rolling window.  Positive excess kurtosis (leptokurtic) indicates
heavy tails and more extreme outliers, while negative excess kurtosis
(platykurtic) indicates light tails.

This module uses Fisher's definition of excess kurtosis (unbiased for normal
distributions), computed via running sums of powers (x, x², x³, x⁴) for
O(1) update per element.

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


@jit(nopython=True, fastmath=True, cache=True)
def _kurtosis_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-compiled core for rolling excess kurtosis (Fisher's definition).

    Uses running sums of powers (sum x, sum x², sum x³, sum x⁴) for O(1)
    update per element.  The formula is the unbiased estimator for normal
    distributions (excess kurtosis = 0 for a normal distribution).

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 4 for a meaningful finite value).

    Returns
    -------
    np.ndarray
        Float64 array of rolling excess kurtosis, with first `length-1`
        elements set to NaN.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial sums for the first window
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for i in range(length):
        x = close[i]
        s1 += x
        s2 += x * x
        s3 += x * x * x
        s4 += x * x * x * x

    def compute_kurtosis(s1, s2, s3, s4, L):  # noqa: N803
        if L < 4:
            return np.nan
        mean = s1 / L
        # Central moments (not normalised)
        M2 = s2 - L * mean * mean  # noqa: N806
        M4 = (  # noqa: N806
            s4
            - 4.0 * mean * s3
            + 6.0 * mean * mean * s2
            - 3.0 * L * mean * mean * mean * mean
        )
        if M2 <= 0.0:
            return np.nan
        # Fisher's excess kurtosis (unbiased for normal)
        return (
            (L * (L + 1) * M4 - 3.0 * (L - 1) * M2 * M2)
            / ((L - 2) * (L - 3) * M2 * M2)
        )

    out[length - 1] = compute_kurtosis(s1, s2, s3, s4, length)
    # Sliding update
    for i in range(length, n):
        add = close[i]
        rem = close[i - length]
        s1 += add - rem
        s2 += add * add - rem * rem
        s3 += add * add * add - rem * rem * rem
        s4 += add * add * add * add - rem * rem * rem * rem
        out[i] = compute_kurtosis(s1, s2, s3, s4, length)
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
    array([       nan,        nan,        nan, -2.       , -2.       , -2.       ])

    """  # noqa: E501
    close = np.asarray(close, dtype=np.float64)
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
    array([       nan,        nan,        nan, -2.       , -2.       , -2.       ])

    """  # noqa: E501
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
    │ 4.0   ┆ -2.0     │
    │ 5.0   ┆ -2.0     │
    │ 6.0   ┆ -2.0     │
    └───────┴──────────┘

    """
    close = df[close_col].to_numpy()
    result = kurtosis_ind(close, length, offset, fillna)
    out_name = output_col or f'KURT_{length}'
    return df.with_columns(pl.Series(out_name, result))
