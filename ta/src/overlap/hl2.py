# -*- coding: utf-8 -*-
"""HL2 (High-Low average) implementation.

HL2 is defined as (high + low) / 2. It is a simple but widely used price
representation in technical analysis, often used as a proxy for the typical
price when close is not available or for certain indicators.

This module provides:
- Numba-accelerated core (`_hl2`)
- Universal wrapper (`hl2_ind`)
- Polars integration (`hl2_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna, replace_inf_with_nan


# ----------------------------------------------------------------------
# Core HL2 function
# ----------------------------------------------------------------------
def _hl2(
    high: np.ndarray,
    low: np.ndarray,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """HL2 (average of high and low) with offset and fillna.

    Parameters
    ----------
    high : np.ndarray
        1D float64 array of high prices.
    low : np.ndarray
        1D float64 array of low prices.
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.

    Returns
    -------
    np.ndarray
        HL2 values, same length as `high` and `low`.

    Notes
    -----
    - Infinites in `high` or `low` are replaced with NaN.
    - All operations are IEEE 754 compliant.

    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    # Replace infinities with NaN
    high = high.copy()
    low = low.copy()
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    avg = (high + low) * 0.5
    return _apply_offset_fillna(avg, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def hl2_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal HL2 (accepts numpy arrays or Polars Series).

    Parameters
    ----------
    high : np.ndarray or pl.Series
        High prices.
    low : np.ndarray or pl.Series
        Low prices.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.

    Returns
    -------
    np.ndarray
        HL2 values, same length as inputs.

    Notes
    -----
    - If inputs are Polars Series, they are converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    return _hl2(high, low, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def hl2_polars(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add HL2 column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col : str, default 'high'
        Name of the column containing high prices.
    low_col : str, default 'low'
        Name of the column containing low prices.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    output_col : str or None, default None
        Name of the output column. If None, defaults to 'HL2'.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with an additional column containing HL2 values.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    result = hl2_ind(high, low, offset, fillna)
    out_name = output_col or 'HL2'
    return df.with_columns([pl.Series(out_name, result)])
