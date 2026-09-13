# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import polars as pl

from numba import njit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)


# ----------------------------------------------------------------------
# Cached weights for ALMA (avoid recomputation)
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _alma_weights(length: int, sigma: float, dist_offset: float) -> np.ndarray:
    """Generate normalized weights for ALMA.

    Parameters
    ----------
    length : int
        Window length.
    sigma : float
        Smoothing factor.
    dist_offset : float
        Distribution offset (0 to 1).

    Returns
    -------
    np.ndarray
        Normalized weights.

    """
    x = np.arange(length, dtype=np.float64)
    k = dist_offset * (length - 1)
    w = np.exp(-0.5 * ((sigma / length) * (x - k)) ** 2)
    w /= w.sum()
    # Protect the lru_cache from accidental in-place modification
    w.flags.writeable = False
    return w


@njit(cache=True)
def _alma_numba_full(
    arr: np.ndarray,
    weights: np.ndarray,
    offset: int,
    fillna: float | None,
) -> np.ndarray:
    """ALMA core with integrated offset and fillna."""
    n = len(arr)
    length = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        # Not enough data - fill with fillna if provided
        if fillna is not None:
            out[:] = fillna
        return out
    # Main ALMA calculation
    for i in range(length - 1, n):
        acc = 0.0
        for j in range(length):
            acc += arr[i - j] * weights[length - 1 - j]
        out[i] = acc
    # Use the universal offset/fillna utility
    return _apply_offset_fillna(out, offset, fillna)


def alma_numba_opt(
    close: np.ndarray,
    length: int = 9,
    sigma: float = 6.0,
    dist_offset: float = 0.85,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Arnaud Legoux Moving Average using Numba (optimized).

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        ALMA period.
    sigma : float
        Smoothing factor.
    dist_offset : float
        Distribution offset (0 to 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        ALMA values.

    Raises
    ------
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - All floating-point operations follow IEEE 754 rules.
    - Infinite values (inf, -inf) are replaced with NaN.
    - NaN values propagate naturally through the calculation.

    """
    if length < 1:
        raise ValueError("ALMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Replace infinities with NaN (IEEE 754 compliance)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")
    # Ensure C-contiguous for best performance
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    weights = _alma_weights(length, sigma, dist_offset)
    return _alma_numba_full(close, weights, offset, fillna)


# ----------------------------------------------------------------------
# Universal ALMA function (TA-Lib not available)
# ----------------------------------------------------------------------
def alma_ind(
    close: np.ndarray | pl.Series,
    length: int = 9,
    sigma: float = 6.0,
    dist_offset: float = 0.85,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal ALMA (always uses Numba)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return alma_numba_opt(
        close, length, sigma, dist_offset, offset, fillna, nan_policy
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def alma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 9,
    sigma: float = 6.0,
    dist_offset: float = 0.85,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """ALMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        ALMA period.
    sigma : float
        Smoothing factor.
    dist_offset : float
        Distribution offset (0 to 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str, optional
        Output column name (default f"ALMA_{length}_{sigma}_{dist_offset}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.

    """
    close = df[close_col].to_numpy()
    result = alma_ind(
        close,
        length=length,
        sigma=sigma,
        dist_offset=dist_offset,
        offset=offset,
        fillna=fillna,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"ALMA_{length}_{sigma}_{dist_offset}"
    return df.with_columns([pl.Series(out_name, result)])
