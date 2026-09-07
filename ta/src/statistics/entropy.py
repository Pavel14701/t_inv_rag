# -*- coding: utf-8 -*-
"""Rolling entropy (ENTP) for financial time series.

Entropy measures the uncertainty or randomness of a distribution.
In the context of financial time series, rolling entropy of returns or
prices can be used as an indicator of market inefficiency or complexity.

This module computes the Shannon entropy of a sliding window of prices,
where the price values are treated as categories and the entropy is
calculated based on the frequency of each unique value in the window.

Functions:
    entropy_numba: Numba-accelerated rolling entropy.
    entropy_ind: Universal rolling entropy (numpy or Polars Series).
    entropy_polars: Add entropy column to Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl
from numba import jit

from .._array_ops import _apply_offset_fillna


@jit(nopython=True, fastmath=False, cache=True)
def _entropy_numba_core(
    close: np.ndarray, length: int, base: float
) -> np.ndarray:
    """Numba-compiled core for rolling Shannon entropy.

    For each window, the values are sorted and the frequency of each
    unique value is counted.  The entropy is computed as:
    -sum(p_i * log(p_i)) where p_i = count_i / length.

    Follows IEEE 754 strictly (``fastmath=False``): a window containing
    non-finite values (NaN, +/-inf) yields NaN for that window; later
    windows recover once the non-finite value leaves the window.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int
        Window size (must be >= 2).
    base : float
        Logarithm base.  Use 2.0 for bits, e (2.71828...) for nats,
        10.0 for dits.

    Returns
    -------
    np.ndarray
        Float64 array of rolling entropy, with first `length-1` elements
        set to NaN.

    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out

    log_base = np.log(base)

    for i in range(length - 1, n):
        # IEEE 754: a window containing non-finite values yields NaN
        finite_window = True
        for j in range(i - length + 1, i + 1):
            if not np.isfinite(close[j]):
                finite_window = False
                break
        if not finite_window:
            continue

        window = close[i - length + 1: i + 1].copy()
        window.sort()

        entropy = 0.0
        j = 0
        while j < length:
            val = window[j]
            cnt = 1
            while j + cnt < length and window[j + cnt] == val:
                cnt += 1
            p = cnt / length
            entropy -= p * np.log(p)
            j += cnt

        out[i] = entropy / log_base

    return out


def entropy_numba(
    close: np.ndarray,
    length: int = 10,
    base: float = 2.0,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Numba-accelerated rolling Shannon entropy.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        Window size (must be >= 2).
    base : float, default 2.0
        Logarithm base (2.0 = bits, e = nats, 10.0 = dits).
    offset : int, default 0
        Shift applied to the output array. Positive = forward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling entropy, shifted and NaN-filled
        according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> entropy_numba(prices, length=3, base=2.0)
    array([       nan,        nan, 1.5849625 , 1.5849625 , 1.5849625 ])

    """
    close = np.asarray(close, dtype=np.float64)
    if length < 2:
        raise ValueError('length must be >= 2')
    if base <= 0.0 or base == 1.0:
        raise ValueError('base must be positive and != 1')
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()

    result = _entropy_numba_core(close, length, base)
    return _apply_offset_fillna(result, offset, fillna)


def entropy_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    base: float = 2.0,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal rolling entropy (numpy array or Polars Series).

    This is a wrapper around :func:`entropy_numba` that automatically converts
    Polars Series to numpy arrays before processing.  All other parameters
    behave exactly as in :func:`entropy_numba`.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        1D array or Polars Series of close prices.
    length : int, default 10
        Window size.
    base : float, default 2.0
        Logarithm base.
    offset : int, default 0
        Shift applied to the output.
    fillna : float or None, default None
        Value to fill NaN positions after offset.

    Returns
    -------
    np.ndarray
        Float64 array of rolling entropy.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return entropy_numba(close, length, base, offset, fillna)


def entropy_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    base: float = 2.0,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """Add a rolling entropy column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 10
        Window size.
    base : float, default 2.0
        Logarithm base.
    offset : int, default 0
        Shift applied to the output column.
    fillna : float or None, default None
        Value to fill NaN positions after offset.
    output_col : str or None, default None
        Name of the output column. If None, uses f'ENTP_{length}'.

    Returns
    -------
    pl.Series
        A new Polars Series containing the rolling entropy.

    """
    close = df[close_col].to_numpy()
    result = entropy_ind(close, length, base, offset, fillna)
    out_name = output_col or f'ENTP_{length}'
    return pl.Series(out_name, result)