# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import njit

from .. import talib, talib_available


# ----------------------------------------------------------------------
# Optimized SMA using Numba (nopython mode, fastmath)
# ----------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def _sma_numba_opt(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Simple Moving Average using Numba.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array.
    length : int
        Window length.

    Returns
    -------
    np.ndarray
        SMA values; first (length-1) positions are NaN.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    cum = np.cumsum(arr)
    out[length - 1:] = (
        cum[length - 1:] - np.concatenate(
            ([0], cum[:n - length])
        )
    ) / length
    return out


# ----------------------------------------------------------------------
# SMA using Numba (with offset and fillna)
# ----------------------------------------------------------------------
def _sma_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Simple Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        SMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        SMA values.
    """
    close = close.astype(np.float64)
    sma = _sma_numba_opt(close, length)
    if offset != 0:
        sma = np.roll(sma, offset)
        if offset > 0:
            sma[:offset] = np.nan
        else:
            sma[offset:] = np.nan
    if fillna is not None:
        sma = np.where(np.isnan(sma), fillna, sma)
    return sma


# ----------------------------------------------------------------------
# SMA using TA-Lib
# ----------------------------------------------------------------------
def sma_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Simple Moving Average via TA-Lib.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        SMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        SMA values.
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    close = close.astype(np.float64)
    sma = talib.SMA(close, timeperiod=length)
    if offset != 0:
        sma = np.roll(sma, offset)
        if offset > 0:
            sma[:offset] = np.nan
        else:
            sma[offset:] = np.nan
    if fillna is not None:
        sma = np.where(np.isnan(sma), fillna, sma)
    return sma


# ----------------------------------------------------------------------
# Universal SMA function (automatic backend selection)
# ----------------------------------------------------------------------
def sma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal SMA with automatic implementation selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        SMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Fill NaN with this value.
    use_talib : bool
        If True and TA-Lib is available, use it; else use Numba.

    Returns
    -------
    np.ndarray
        SMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = close.astype(np.float64)
    if use_talib and talib_available:
        return sma_talib(close, length, offset, fillna)
    else:
        return _sma_numba(close, length, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def sma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    SMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        SMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA-Lib if available.
    output_col : str, optional
        Output column name (default f"SMA_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.
    """
    close = df[close_col].to_numpy()
    result = sma_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib
    )
    out_name = output_col or f"SMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])