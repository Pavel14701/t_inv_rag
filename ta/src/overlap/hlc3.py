# -*- coding: utf-8 -*-
"""HLC3 (High-Low-Close average) implementation.

HLC3 is defined as (high + low + close) / 3. It is a common price
representation used in technical analysis as a proxy for the typical price.

This module provides:
- Numba-accelerated core (`_hlc3`)
- Universal wrapper (`hlc3_ind`)
- Polars integration (`hlc3_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna, replace_inf_with_nan


# ----------------------------------------------------------------------
# Core HLC3 function
# ----------------------------------------------------------------------
def _hlc3(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """HLC3 (average of high, low and close) with offset and fillna.

    Parameters
    ----------
    high : np.ndarray
        1D float64 array of high prices.
    low : np.ndarray
        1D float64 array of low prices.
    close : np.ndarray
        1D float64 array of close prices.
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.

    Returns
    -------
    np.ndarray
        HLC3 values, same length as input arrays.

    Notes
    -----
    - Infinites in `high`, `low`, or `close` are replaced with NaN.
    - All operations are IEEE 754 compliant.

    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Replace infinities with NaN
    high = high.copy()
    low = low.copy()
    close = close.copy()
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    replace_inf_with_nan(close)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    avg = (high + low + close) / 3.0
    return _apply_offset_fillna(avg, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def hlc3_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal HLC3 (accepts numpy arrays or Polars Series).

    Parameters
    ----------
    high : np.ndarray or pl.Series
        High prices.
    low : np.ndarray or pl.Series
        Low prices.
    close : np.ndarray or pl.Series
        Close prices.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.

    Returns
    -------
    np.ndarray
        HLC3 values, same length as inputs.

    Notes
    -----
    - If inputs are Polars Series, they are converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return _hlc3(high, low, close, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def hlc3_polars(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add HLC3 column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col : str, default 'high'
        Name of the column containing high prices.
    low_col : str, default 'low'
        Name of the column containing low prices.
    close_col : str, default 'close'
        Name of the column containing close prices.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    output_col : str or None, default None
        Name of the output column. If None, defaults to 'HLC3'.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with an additional column containing HLC3 values.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = hlc3_ind(high, low, close, offset, fillna)
    out_name = output_col or 'HLC3'
    return df.with_columns([pl.Series(out_name, result)])
