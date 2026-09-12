# -*- coding: utf-8 -*-
"""Triple Exponential Moving Average (TEMA) implementation.

TEMA is defined as: TEMA = 3*(EMA(close) - EMA(EMA(close))) + EMA(EMA(EMA(close))).

This module provides:
- Numba-accelerated fallback implementation
- TA-Lib backend (if available)
- Universal wrapper (`tema_ind`)
- Polars integration (`tema_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""  # noqa: E501

import numpy as np
import polars as pl

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available
from .ema import _ema_numba_opt


def tema_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Triple Exponential Moving Average using Numba (fallback backend).

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of closing prices.
    length : int, default 10
        TEMA period.
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        TEMA values, same length as `close`.

    Raises
    ------
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - The first `3*(length-1)` elements may be NaN due to the triple EMA.
    - Each nested EMA is seeded on the valid (non-warmup) part of the
      previous one; otherwise the NaN warmup head would poison the result.
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError("TEMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64, copy=False)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")

    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    n = len(close)
    valid_start = length - 1
    ema1 = _ema_numba_opt(close, length)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    ema3 = np.full(n, np.nan, dtype=np.float64)
    if n > valid_start:
        # EMA of EMA: seed on the valid part of ema1 (skip the NaN warmup).
        ema2_tail = _ema_numba_opt(
            np.ascontiguousarray(ema1[valid_start:]), length
        )
        ema2[2 * valid_start :] = ema2_tail[valid_start:]
        if n > 2 * valid_start:
            # Third EMA: seed on the valid part of ema2.
            ema3_tail = _ema_numba_opt(
                np.ascontiguousarray(ema2[2 * valid_start :]), length
            )
            ema3[3 * valid_start :] = ema3_tail[valid_start:]
    tema = 3.0 * (ema1 - ema2) + ema3
    return _apply_offset_fillna(tema, offset, fillna)


# ----------------------------------------------------------------------
# TEMA using TA-Lib (if available)
# ----------------------------------------------------------------------
def tema_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Triple Exponential Moving Average via TA-Lib.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of closing prices.
    length : int, default 10
        TEMA period.
    offset : int, default 0
        Shift the result. Positive = forward, negative = backward.
    fillna : float or None, default None
        Value to replace NaN and shifted-in positions. If None, NaN remains.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        TEMA values, same length as `close`.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    if length < 1:
        raise ValueError("TEMA length must be >= 1")

    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")

    tema = talib.TEMA(close, timeperiod=length)
    return _apply_offset_fillna(tema, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def tema_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal TEMA with backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 10
        EMA period.
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    use_talib : bool, default True
        If True and TA-Lib is available, use it; else use Numba.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        TEMA values.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = np.asarray(close, dtype=np.float64)

    if use_talib and talib_available:
        return tema_talib(close, length, offset, fillna, nan_policy)
    else:
        return tema_numba(close, length, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def tema_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add TEMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, offset, fillna, use_talib : as above.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str, optional
        Output column name (default f"TEMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with TEMA column.

    """
    close = df[close_col].to_numpy()
    result = tema_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"TEMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])
