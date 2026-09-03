"""Rolling quantile (QTL) for financial time series.

This module provides Numba-accelerated computation of rolling quantiles
(e.g., median, lower/upper quartiles, or any other percentile).  The
quantile is computed by sorting each sliding window and picking the
value at the desired position.

The implementation supports any quantile q in (0, 1) and uses a simple
rounding method for the index: idx = round(q * (length - 1)).

Functions:
    quantile_numba: Numba-accelerated rolling quantile.
    quantile_ind: Universal rolling quantile (numpy or Polars Series).
    quantile_polars: Add quantile column to Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl
from numba import jit

from .._array_ops import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _quantile_numba_core(
    close: np.ndarray,
    length: int,
    q: float
) -> np.ndarray:
    """Numba-compiled core for rolling quantile.

    For each window, the values are sorted and the quantile is selected
    using the index `int(round(q * (length - 1)))`.  This is equivalent
    to the "linear" interpolation method in many statistical packages.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 2).
    q : float
        Quantile value, between 0 and 1 inclusive.  For example:
        - 0.5  : median
        - 0.25 : lower quartile
        - 0.75 : upper quartile

    Returns
    -------
    np.ndarray
        Float64 array of rolling quantiles, with first `length-1` elements
        set to NaN.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out

    # Pre-compute the index based on the quantile and window length
    idx = int(round(q * (length - 1)))

    for i in range(length - 1, n):
        window = close[i - length + 1: i + 1].copy()  # copy for sorting
        window.sort()
        out[i] = window[idx]

    return out


def quantile_numba(
    close: np.ndarray,
    length: int = 30,
    q: float = 0.5,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numba-accelerated rolling quantile.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size (must be >= 2).
    q : float, default 0.5
        Quantile value, between 0 and 1 inclusive.
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling quantiles, shifted and NaN-filled
        according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> quantile_numba(prices, length=3, q=0.5)  # median
    array([       nan,        nan, 2.       , 3.       , 4.       , 5.       ])

    """
    close = np.asarray(close, dtype=np.float64)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    result = _quantile_numba_core(close, length, q)
    return _apply_offset_fillna(result, offset, fillna)


def quantile_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    q: float = 0.5,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal rolling quantile (numpy array or Polars Series).

    This is a wrapper around :func:`quantile_numba` that automatically converts
    Polars Series to numpy arrays before processing.  All other parameters
    behave exactly as in :func:`quantile_numba`.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        1D array or Polars Series of close prices.
    length : int, default 30
        Window size.
    q : float, default 0.5
        Quantile value (0 < q < 1).
    offset : int, default 0
        Shift applied to the output.
    fillna : float or None, default None
        Value to fill NaN positions after offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling quantiles.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> quantile_ind(s, length=3, q=0.5)
    array([       nan,        nan, 2.       , 3.       , 4.       , 5.       ])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return quantile_numba(close, length, q, offset, fillna)


def quantile_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 30,
    q: float = 0.5,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """Add a rolling quantile column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 30
        Window size.
    q : float, default 0.5
        Quantile value (0 < q < 1).
    offset : int, default 0
        Shift applied to the output column.
    fillna : float or None, default None
        Value to fill NaN positions after offset.
    output_col : str or None, default None
        Name of the output column. If None, uses f'QTL_{length}_{q}'.

    Returns
    -------
    pl.Series
        A new Polars Series containing the rolling quantile.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    >>> quantile_polars(df, length=3, q=0.5)
    shape: (6,)
    Series: 'QTL_3_0.5' [f64]
    [
        null
        null
        2.0
        3.0
        4.0
        5.0
    ]

    """
    close = df[close_col].to_numpy()
    result = quantile_ind(close, length, q, offset, fillna)
    out_name = output_col or f'QTL_{length}_{q}'
    return pl.Series(out_name, result)
