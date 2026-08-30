# -*- coding: utf-8 -*-
"""Simple Moving Average (SMA) for financial time series.

Provides Numba-accelerated and TA-Lib implementations.
"""

import numpy as np
import polars as pl
from numba import njit

from ..external import talib, talib_available
from .._array_ops import _apply_offset_fillna, replace_inf_with_nan


@njit('float64[:](float64[:], int64)', cache=True)
def _sma_numba_opt(arr: np.ndarray, length: int) -> np.ndarray:
    """Numba-accelerated core for SMA using cumulative sum.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices (assumed to have no NaNs or infinities).
    length : int
        Window size (must be >= 1).

    Returns
    -------
    np.ndarray
        SMA array with first `length-1` elements set to NaN.
        If `len(arr) < length`, returns all NaN.

    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    cum = np.cumsum(arr)
    out[length - 1:] = (
        cum[length - 1:] - np.concatenate((np.array([0.0]), cum[:n - length]))
    ) / length
    return out


def _sma_numba(  # noqa: C901
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """SMA using Numba with NaN handling, offset, fillna, and trim.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        Window size (must be >= 1).
    offset : int, default 0
        Shift applied to the result. Positive = forward.
    fillna : float or None, default None
        Value to replace NaN after shift.
    nan_policy : str, default 'raise'
        How to handle NaNs in input:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.
    trim : bool, default False
        If True, remove the first `length-1` elements.
        Incompatible with `offset`.

    Returns
    -------
    np.ndarray
        SMA values, shifted and NaN-filled as requested.

    Raises
    ------
    ValueError
        If length < 1, input series too short, invalid nan_policy,
        or offset and trim are used together.

    Notes
    -----
    - Infinites are replaced with NaN before any calculation.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError('SMA length must be >= 1')
    close = np.asarray(close, dtype=np.float64)
    if len(close) < length:
        raise ValueError(
            f'Input series too short: need at least {length} elements, '
            f'got {len(close)}.'
        )

    # Replace infinities with NaN
    close = close.copy()
    replace_inf_with_nan(close)

    # NaN handling
    if np.isnan(close).any():
        if nan_policy == 'raise':
            raise ValueError(
                'Input contains NaN values.',
                "Use nan_policy='ffill', 'bfill' or 'both'."
            )
        close = close.copy()
        if nan_policy == 'ignore':
            pass
        elif nan_policy == 'ffill':
            for i in range(1, len(close)):
                if np.isnan(close[i]):
                    close[i] = close[i - 1]
        elif nan_policy == 'bfill':
            for i in range(len(close) - 2, -1, -1):
                if np.isnan(close[i]):
                    close[i] = close[i + 1]
        elif nan_policy == 'both':
            for i in range(1, len(close)):
                if np.isnan(close[i]):
                    close[i] = close[i - 1]
            for i in range(len(close) - 2, -1, -1):
                if np.isnan(close[i]):
                    close[i] = close[i + 1]
        else:
            raise ValueError(
                f'Unknown nan_policy: {nan_policy}. '
                "Use 'raise', 'ignore', 'ffill', 'bfill', or 'both'."
            )
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    sma = _sma_numba_opt(close, length)
    if trim:
        if offset != 0:
            raise ValueError('offset and trim cannot be used simultaneously.')
        sma = sma[length - 1:]
    return _apply_offset_fillna(sma, offset, fillna)


def sma_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """SMA via TA-Lib.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 10
        Window size.
    offset : int, default 0
        Shift applied to the result.
    fillna : float or None, default None
        Value to replace NaN after shift.

    Returns
    -------
    np.ndarray
        SMA values.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.

    Notes
    -----
    - TA-Lib does not handle NaNs; input must be clean.
    - Infinites are replaced with NaN before calculation.

    """
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    sma = talib.SMA(close, timeperiod=length)
    return _apply_offset_fillna(sma, offset, fillna)


def sma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Universal SMA with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        1D array of close prices.
    length : int, default 10
        Window size.
    offset : int, default 0
        Shift applied to the result.
    fillna : float or None, default None
        Value to fill NaN after shift.
    use_talib : bool, default True
        If True and TA-Lib is available, use TA-Lib.
    nan_policy : str, default 'raise'
        How to handle NaNs (only for Numba backend).
    trim : bool, default False
        If True, remove first `length-1` elements. Incompatible with TA-Lib.

    Returns
    -------
    np.ndarray
        SMA values.

    Raises
    ------
    ValueError
        If trim=True and TA-Lib backend is selected.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    >>> sma_ind(prices, length=3, use_talib=False)
    array([       nan,        nan, 2.        , 3.        , 4.        ,
            5.        , 6.        , 7.        , 8.        , 9.        ])
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> sma_ind(s, length=3, use_talib=False, trim=True)
    array([2., 3., 4.])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = np.asarray(close, dtype=np.float64)

    if use_talib and talib_available:
        if trim:
            raise ValueError('trim=True is not supported with TA-Lib backend.')
        return sma_talib(close, length, offset, fillna)
    else:
        return _sma_numba(close, length, offset, fillna, nan_policy, trim)


def sma_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None,
) -> pl.Series:
    """Return SMA as a Polars Series.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    length : int, default 10
        Window size.
    offset : int, default 0
        Shift applied to the result.
    fillna : float or None, default None
        Value to fill NaN after shift.
    use_talib : bool, default True
        Use TA-Lib if available.
    nan_policy : str, default 'raise'
        NaN handling policy (only for Numba backend).
    output_col : str or None, default None
        Name of the output Series. If None, uses f'SMA_{length}'.

    Returns
    -------
    pl.Series
        SMA values.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    >>> sma_polars(df, length=3, output_col='SMA3')
    shape: (10,)
    Series: 'SMA3' [f64]
    [
        null
        null
        2.0
        3.0
        4.0
        5.0
        6.0
        7.0
        8.0
        9.0
    ]

    """  # noqa: E501
    close = df[close_col].to_numpy()
    result = sma_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,
    )
    out_name = output_col or f'SMA_{length}'
    return pl.Series(out_name, result)
