"""Rolling median (MEDIAN) for financial time series.

This module provides Numba-accelerated computation of rolling medians
using partial sort (NumPy's partition) for O(n log k) performance instead
of full O(n log n) sorting.  The median is the middle value of a window
when sorted; for even window sizes, the average of the two middle values
is returned.

The implementation supports both odd and even window sizes and is
optimised for speed.

Functions:
    median_numba: Numba-accelerated rolling median.
    median_ind: Universal rolling median (numpy or Polars Series).
    median_polars: Add median column to Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import _apply_offset_fillna


@jit(nopython=True, cache=True)
def _window_is_finite(window: np.ndarray) -> bool:
    """Return True if every element of `window` is finite (no NaN/inf)."""
    for j in range(len(window)):
        if not np.isfinite(window[j]):
            return False
    return True


@jit(nopython=True, fastmath=False, cache=True)
def _median_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    """Numba-compiled core for rolling median.

    Uses NumPy's partition to find the median without fully sorting the
    window.  For odd length, the middle element is selected directly.
    For even length, the average of the two middle elements is computed.
    Windows containing NaN/inf yield NaN: without the explicit check,
    partition() would silently place the non-finite value at the end and
    return a finite median for a window that contains it.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 1).

    Returns
    -------
    np.ndarray
        Float64 array of rolling medians, with first `length-1` elements
        set to NaN.  NaN/inf inputs propagate to windows containing them.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out

    if length % 2 == 1:
        # Odd window: single middle element
        kth = length // 2
        for i in range(length - 1, n):
            window = close[i - length + 1 : i + 1].copy()
            if not _window_is_finite(window):
                continue  # keep NaN (out is pre-filled with NaN)
            part = np.partition(window, kth)
            out[i] = part[kth]
    else:
        # Even window: average of two middle elements
        kth1 = length // 2 - 1
        kth2 = length // 2
        for i in range(length - 1, n):
            window = close[i - length + 1 : i + 1].copy()
            if not _window_is_finite(window):
                continue  # keep NaN (out is pre-filled with NaN)
            part = np.partition(window, [kth1, kth2])
            out[i] = (part[kth1] + part[kth2]) * 0.5

    return out


def median_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numba-accelerated rolling median.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window size (must be >= 1).
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling medians, shifted and NaN-filled
        according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> median_numba(prices, length=3)
    array([       nan,        nan, 2.       , 3.       , 4.       , 5.       ])

    """
    close = np.asarray(close, dtype=np.float64)
    if length < 1:
        raise ValueError("length must be >= 1")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()

    result = _median_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def median_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal rolling median (numpy array or Polars Series).

    This is a wrapper around :func:`median_numba` that automatically converts
    Polars Series to numpy arrays before processing.  All other parameters
    behave exactly as in :func:`median_numba`.

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
        Float64 array of rolling medians.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> median_ind(s, length=3)
    array([       nan,        nan, 2.       , 3.       , 4.       , 5.       ])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return median_numba(close, length, offset, fillna)


def median_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """Add a rolling median column to a Polars DataFrame.

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
        Name of the output column. If None, uses f'MEDIAN_{length}'.

    Returns
    -------
    pl.Series
        A new Polars Series containing the rolling median.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    >>> median_polars(df, length=3)
    shape: (6,)
    Series: 'MEDIAN_3' [f64]
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
    result = median_ind(close, length, offset, fillna)
    out_name = output_col or f"MEDIAN_{length}"
    return pl.Series(out_name, result)
