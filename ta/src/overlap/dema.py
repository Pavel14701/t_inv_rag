# -*- coding: utf-8 -*-
"""Double Exponential Moving Average (DEMA) implementation.

DEMA is defined as: DEMA = 2 * EMA(close, length) - EMA(EMA(close, length), length).

This module provides:
- Numba-accelerated fallback implementation
- TA-Lib backend (if available)
- Universal wrapper (`dema_ind`)
- Polars integration (`dema_polars`)

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


def dema_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Double Exponential Moving Average using Numba (fallback backend).

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of closing prices.
    length : int, default 10
        DEMA period.
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
        DEMA values, same length as `close`.

    Raises
    ------
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - The first `2*(length-1)` elements may be NaN due to the double EMA.
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError('DEMA length must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')

    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    ema1 = _ema_numba_opt(close, length)
    # EMA of EMA: seed on the valid (non-NaN) part of ema1.
    # Seeding on the raw ema1 (which has NaN warmup) would poison
    # ema2 with NaN forever, making DEMA all-NaN.
    n = len(ema1)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    valid_start = length - 1
    ema2_tail = _ema_numba_opt(
        np.ascontiguousarray(ema1[valid_start:]), length
    )
    ema2[2 * valid_start:] = ema2_tail[valid_start:]
    dema = 2.0 * ema1 - ema2
    return _apply_offset_fillna(dema, offset, fillna)


def dema_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Double Exponential Moving Average via TA-Lib.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of closing prices.
    length : int, default 10
        DEMA period.
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
        DEMA values, same length as `close`.

    Raises
    ------
    ImportError
        If TA-Lib is not available.
    ValueError
        If `length < 1`, the input contains NaN with `nan_policy='raise'`,
        or `nan_policy` is unknown.

    Notes
    -----
    - TA-Lib uses a slightly different initialisation method, so results may
        differ from the Numba version for the first few periods.
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    if length < 1:
        raise ValueError('DEMA length must be >= 1')

    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')

    dema = talib.DEMA(close, timeperiod=length)
    return _apply_offset_fillna(dema, offset, fillna)


def dema_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Universal DEMA with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Input price data.
    length : int, default 10
        DEMA period.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    use_talib : bool, default True
        If True and TA-Lib is installed, use TA-Lib.
        Otherwise, fall back to the Numba implementation.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        DEMA values, same length as `close`.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = close.astype(np.float64)

    if use_talib and talib_available:
        return dema_talib(close, length, offset, fillna, nan_policy)
    else:
        return dema_numba(close, length, offset, fillna, nan_policy)


def dema_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add DEMA column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 10
        DEMA period.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    use_talib : bool, default True
        If True and TA-Lib is installed, use TA-Lib.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str or None, default None
        Name of the output column. If None, defaults to f'DEMA_{length}'.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with an additional column containing DEMA values.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = dema_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    output_name = output_col or f'DEMA_{length}'
    return df.with_columns([pl.Series(output_name, result)])
