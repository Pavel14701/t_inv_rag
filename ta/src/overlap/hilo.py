# -*- coding: utf-8 -*-
"""HiLo Activator indicator implementation.

The HiLo Activator is a trend-following indicator that uses moving averages of
high and low prices to determine market direction. It returns three series:
- HILO: the activator line (long or short)
- HILOl: long signal (low_ma when in uptrend)
- HILOs: short signal (high_ma when in downtrend)

This module provides:
- Numba-accelerated core logic
- Unified MA backend (Numba or TA-Lib)
- Universal wrapper (`hilo_ind`)
- Polars integration (`hilo_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

import numpy as np
import polars as pl
from numba import jit

from ..external import talib_available
from .._array_ops import _apply_offset_fillna, replace_inf_with_nan
from ..overlap.ema import ema_ind
from ..overlap.sma import sma_ind


def ma_numba(
    arr: np.ndarray,
    length: int,
    mamode: str = 'sma',
    nan_policy: str = 'ignore',
) -> np.ndarray:
    """Unified moving average using Numba backend.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices.
    length : int
        Moving average period.
    mamode : str, default 'sma'
        Type of moving average: 'sma' or 'ema'.
    nan_policy : str, default 'ignore'
        How to handle NaN values (passed to the underlying indicator).

    Returns
    -------
    np.ndarray
        Moving average values, same length as `arr`.

    Raises
    ------
    ValueError
        If `mamode` is not supported.

    """
    arr = arr.astype(np.float64)
    mamode = mamode.lower()
    if mamode == 'sma':
        return sma_ind(arr, length, use_talib=False, nan_policy=nan_policy)
    elif mamode == 'ema':
        return ema_ind(arr, length, use_talib=False, nan_policy=nan_policy)
    else:
        raise ValueError(f'Unsupported mamode: {mamode}')


def ma_talib(
    arr: np.ndarray,
    length: int,
    mamode: str = 'sma',
    nan_policy: str = 'ignore',
) -> np.ndarray:
    """Unified moving average using TA-Lib.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices.
    length : int
        Moving average period.
    mamode : str, default 'sma'
        Type of moving average: 'sma' or 'ema'.
    nan_policy : str, default 'ignore'
        How to handle NaN values (passed to the underlying indicator).

    Returns
    -------
    np.ndarray
        Moving average values, same length as `arr`.

    Raises
    ------
    ImportError
        If TA-Lib is not available.
    ValueError
        If `mamode` is not supported.

    """
    if not talib_available:
        raise ImportError('TA-Lib not available')
    arr = arr.astype(np.float64)
    mamode = mamode.lower()
    if mamode == 'sma':
        return sma_ind(arr, length, use_talib=True, nan_policy=nan_policy)
    elif mamode == 'ema':
        return ema_ind(arr, length, use_talib=True, nan_policy=nan_policy)
    else:
        raise ValueError(f'Unsupported mamode: {mamode}')


def ma(
    arr: np.ndarray,
    length: int,
    mamode: str = 'sma',
    use_talib: bool = True,
    nan_policy: str = 'ignore',
) -> np.ndarray:
    """Universal moving average with automatic backend selection.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices.
    length : int
        Moving average period.
    mamode : str, default 'sma'
        Type of moving average: 'sma' or 'ema'.
    use_talib : bool, default True
        If True and TA-Lib is installed, use TA-Lib.
    nan_policy : str, default 'ignore'
        How to handle NaN values.

    Returns
    -------
    np.ndarray
        Moving average values.

    Notes
    -----
    - This function is IEEE 754 compliant (via `nan_policy='ignore'`).

    """
    if use_talib and talib_available:
        return ma_talib(arr, length, mamode, nan_policy)
    else:
        return ma_numba(arr, length, mamode, nan_policy)


# ----------------------------------------------------------------------
# Core HiLo logic (Numba)
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _hilo_numba_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    high_ma: np.ndarray,
    low_ma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated core logic for HiLo Activator.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64).
    high_ma, low_ma : np.ndarray
        Pre-computed moving averages of high and low.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (hilo, long, short) – all arrays have the same length as `close`.
        The first element is NaN; subsequent elements follow the rules:
        - if close[i] > high_ma[i-1]  -> hilo[i] = low_ma[i], long = low_ma[i]
        - if close[i] < low_ma[i-1]   -> hilo[i] = high_ma[i], short = high_ma[i]
        - else                        -> hilo[i] = hilo[i-1], long & short = hilo[i-1]

    """
    n = len(close)
    hilo = np.full(n, np.nan, dtype=np.float64)
    long_arr = np.full(n, np.nan, dtype=np.float64)
    short_arr = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return hilo, long_arr, short_arr

    for i in range(1, n):
        if close[i] > high_ma[i - 1]:
            hilo[i] = low_ma[i]
            long_arr[i] = low_ma[i]
            short_arr[i] = np.nan
        elif close[i] < low_ma[i - 1]:
            hilo[i] = high_ma[i]
            short_arr[i] = high_ma[i]
            long_arr[i] = np.nan
        else:
            hilo[i] = hilo[i - 1]
            long_arr[i] = hilo[i - 1]
            short_arr[i] = hilo[i - 1]

    return hilo, long_arr, short_arr


# ----------------------------------------------------------------------
# HiLo with Numba backend
# ----------------------------------------------------------------------
def _hilo_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = 'sma',
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HiLo Activator using Numba for both MA and core logic.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64).
    high_length, low_length : int, default 13, 21
        Periods for high and low moving averages.
    mamode : str, default 'sma'
        Type of moving average ('sma' or 'ema').
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (hilo, long, short) – all arrays have the same length as `close`.

    Notes
    -----
    - If `close` is empty, returns three empty arrays.
    - If `len(close) < min(high_length, low_length)`, returns arrays of NaN.
    - Infinites are replaced with NaN before calculation.
    - All operations are IEEE 754 compliant.

    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)

    n = len(close)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    if n < min(high_length, low_length):
        return (
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
        )

    high = high.copy()
    low = low.copy()
    close = close.copy()
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    replace_inf_with_nan(close)

    high_ma = ma_numba(high, high_length, mamode, nan_policy='ignore')
    low_ma = ma_numba(low, low_length, mamode, nan_policy='ignore')

    hilo, long_arr, short_arr = _hilo_numba_core(high, low, close, high_ma, low_ma)

    hilo = _apply_offset_fillna(hilo, offset, fillna)
    long_arr = _apply_offset_fillna(long_arr, offset, fillna)
    short_arr = _apply_offset_fillna(short_arr, offset, fillna)
    return hilo, long_arr, short_arr


# ----------------------------------------------------------------------
# HiLo with TA-Lib backend for MA
# ----------------------------------------------------------------------
def _hilo_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = 'sma',
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HiLo Activator using TA-Lib for MA and Numba for core logic.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64).
    high_length, low_length : int, default 13, 21
        Periods for high and low moving averages.
    mamode : str, default 'sma'
        Type of moving average ('sma' or 'ema').
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (hilo, long, short) – all arrays have the same length as `close`.

    Raises
    ------
    ImportError
        If TA-Lib is not available.

    Notes
    -----
    - If `close` is empty, returns three empty arrays.
    - If `len(close) < min(high_length, low_length)`, returns arrays of NaN.
    - Infinites are replaced with NaN before calculation.
    - All operations are IEEE 754 compliant.

    """
    if not talib_available:
        raise ImportError('TA-Lib not available')

    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)

    n = len(close)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    if n < min(high_length, low_length):
        return (
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
        )

    high = high.copy()
    low = low.copy()
    close = close.copy()
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    replace_inf_with_nan(close)

    high_ma = ma_talib(high, high_length, mamode, nan_policy='ignore')
    low_ma = ma_talib(low, low_length, mamode, nan_policy='ignore')

    hilo, long_arr, short_arr = _hilo_numba_core(high, low, close, high_ma, low_ma)

    hilo = _apply_offset_fillna(hilo, offset, fillna)
    long_arr = _apply_offset_fillna(long_arr, offset, fillna)
    short_arr = _apply_offset_fillna(short_arr, offset, fillna)
    return hilo, long_arr, short_arr


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def hilo_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = 'sma',
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Universal HiLo Activator indicator.

    Parameters
    ----------
    high, low, close : np.ndarray or pl.Series
        Price data.
    high_length, low_length : int, default 13, 21
        Periods for high and low moving averages.
    mamode : str, default 'sma'
        Moving average type: 'sma' or 'ema'.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    use_talib : bool, default True
        If True and TA-Lib is installed, use TA-Lib for MA.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (hilo, long, short) – numpy arrays of float64.

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

    if use_talib and talib_available:
        return _hilo_talib(
            high, low, close, high_length, low_length, mamode, offset, fillna
        )
    else:
        return _hilo_numba(
            high, low, close, high_length, low_length, mamode, offset, fillna
        )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def hilo_polars(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = 'sma',
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = '',
) -> pl.DataFrame:
    """Add HiLo Activator columns to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col, low_col, close_col : str, default 'high', 'low', 'close'
        Names of columns with high, low, and close prices.
    high_length, low_length : int, default 13, 21
        Periods for high and low moving averages.
    mamode : str, default 'sma'
        Moving average type: 'sma' or 'ema'.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to replace NaNs.
    use_talib : bool, default True
        If True and TA-Lib is installed, use TA-Lib for MA.
    suffix : str, default ''
        Custom suffix for column names. If empty, default suffix
        f'_{high_length}_{low_length}' is used.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with three new columns:
        - HILO{suffix}
        - HILOl{suffix}
        - HILOs{suffix}

    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()

    hilo_arr, long_arr, short_arr = hilo_ind(
        high, low, close,
        high_length=high_length,
        low_length=low_length,
        mamode=mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )

    if not suffix:
        suffix = f'_{high_length}_{low_length}'

    return df.with_columns([
        pl.Series(f'HILO{suffix}', hilo_arr),
        pl.Series(f'HILOl{suffix}', long_arr),
        pl.Series(f'HILOs{suffix}', short_arr),
    ])