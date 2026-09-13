# -*- coding: utf-8 -*-
"""Hull Moving Average (HMA) implementation.

The Hull Moving Average reduces lag while maintaining smoothness. It is
calculated as: HMA = MA(2 * MA(close, length/2) - MA(close, length),
    sqrt(length)).

This module provides:
- Numba-accelerated implementation with configurable base MA
- Universal wrapper (`hma_ind`)
- Polars integration (`hma_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna, replace_inf_with_nan
from ..overlap.ema import ema_ind
from ..overlap.sma import sma_ind
from ..overlap.wma import wma_ind


# ----------------------------------------------------------------------
# HMA - Hull Moving Average
# ----------------------------------------------------------------------
def hma_numba(
    close: np.ndarray,
    length: int = 10,
    mamode: str = "wma",
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Hull Moving Average using Numba and selected base MA.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        HMA period (must be >= 2).
    mamode : str, default 'wma'
        Type of moving average for internal calculations:
        'sma', 'ema', or 'wma'.
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.

    Returns
    -------
    np.ndarray
        HMA values, same length as `close`.

    Raises
    ------
    ValueError
        If `length < 2`, `mamode` is not supported, or series too short.

    Notes
    -----
    - The first `int(sqrt(length)) + int(length/2) - 1` elements are NaN.
    - Infinites in `close` are replaced with NaN.
    - All internal MA calls use Numba backend with `nan_policy='ignore'`.

    """
    if length < 2:
        raise ValueError("HMA length must be >= 2")

    if mamode not in ("sma", "ema", "wma"):
        raise ValueError(f"Unsupported mamode: {mamode}")

    close = np.asarray(close, dtype=np.float64, copy=False)
    close = close.copy()
    replace_inf_with_nan(close)

    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))

    if len(close) < max(half_length, length, sqrt_length):
        return np.full(len(close), np.nan, dtype=np.float64)

    # Compute moving averages using wrapper functions
    if mamode == "sma":
        maf = sma_ind(
            close,
            half_length,
            use_talib=False,
            nan_policy="ignore",
            trim=False,
        )
        mas = sma_ind(
            close, length, use_talib=False, nan_policy="ignore", trim=False
        )
        diff = 2.0 * maf - mas
        hma = sma_ind(
            diff, sqrt_length, use_talib=False, nan_policy="ignore", trim=False
        )
    elif mamode == "ema":
        maf = ema_ind(
            close,
            half_length,
            use_talib=False,
            nan_policy="ignore",
            trim=False,
        )
        mas = ema_ind(
            close, length, use_talib=False, nan_policy="ignore", trim=False
        )
        diff = 2.0 * maf - mas
        hma = ema_ind(
            diff, sqrt_length, use_talib=False, nan_policy="ignore", trim=False
        )
    else:  # mamode == 'wma'
        maf = wma_ind(
            close,
            half_length,
            asc=True,
            use_talib=False,
            nan_policy="ignore",
            trim=False,
        )
        mas = wma_ind(
            close,
            length,
            asc=True,
            use_talib=False,
            nan_policy="ignore",
            trim=False,
        )
        diff = 2.0 * maf - mas
        hma = wma_ind(
            diff,
            sqrt_length,
            asc=True,
            use_talib=False,
            nan_policy="ignore",
            trim=False,
        )

    return _apply_offset_fillna(hma, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def hma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    mamode: str = "wma",
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Universal Hull Moving Average (accepts numpy array or Polars Series).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 10
        HMA period.
    mamode : str, default 'wma'
        Base moving average type: 'sma', 'ema', or 'wma'.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.

    Returns
    -------
    np.ndarray
        HMA values, same length as input.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return hma_numba(close, length, mamode, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def hma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    mamode: str = "wma",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add HMA column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 10
        HMA period.
    mamode : str, default 'wma'
        Base moving average type: 'sma', 'ema', or 'wma'.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    output_col : str or None, default None
        Name of the output column. If None, defaults to f'HMA_{length}'.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with an additional column containing HMA values.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = hma_ind(close, length, mamode, offset, fillna)
    out_name = output_col or f"HMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])
