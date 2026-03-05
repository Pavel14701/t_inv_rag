# -*- coding: utf-8 -*-
from typing import cast

import numpy as np
import polars as pl

from .. import _TALIB_MA_MAP, ma_mode, talib, talib_available
from ..utils import _apply_offset_fillna, _rolling_max_numba, _rolling_min_numba


# ----------------------------------------------------------------------
# Core calculation
# ----------------------------------------------------------------------
def stochf_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k: int = 14,
    d: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast Stochastic using Numba rolling min/max and universal MA.

    Returns (stoch_k, stoch_d) as numpy arrays.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    if use_talib and talib_available:
        # TA‑Lib STOCHF accepts fastd_matype parameter
        ma_type = cast(
            talib.MA_Type, 
            _TALIB_MA_MAP.get(mamode.lower(), talib.MA_Type.SMA)
        )
        k_arr, d_arr = talib.STOCHF(
            high, low, close,
            fastk_period=k,
            fastd_period=d,
            fastd_matype=ma_type
        )
        stoch_k = k_arr
        stoch_d = d_arr
    else:
        # Rolling min of low and max of high using Numba
        lowest_low = _rolling_min_numba(low, k)
        highest_high = _rolling_max_numba(high, k)
        denom = highest_high - lowest_low
        with np.errstate(divide='ignore', invalid='ignore'):
            stoch_k = 100.0 * (close - lowest_low) / denom
        # %D is a moving average of %K
        stoch_d = cast(np.ndarray, ma_mode(
            mamode, stoch_k, length=d, offset=0, fillna=None, use_talib=False
        ))
    # Apply global offset and fillna
    stoch_k = _apply_offset_fillna(stoch_k, offset, fillna)
    stoch_d = _apply_offset_fillna(stoch_d, offset, fillna)
    return stoch_k, stoch_d


def stochf_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    k: int = 14,
    d: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal Fast Stochastic (accepts numpy arrays or Polars Series).

    Returns (stoch_k, stoch_d) as numpy arrays.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return stochf_numpy(high, low, close, k, d, mamode, offset, fillna, use_talib)


def stochf_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    k: int = 14,
    d: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Compute Fast Stochastic columns (%K and %D) and return a new DataFrame
    containing the date column and the two indicator columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Names of columns containing high, low and close prices.
    date_col : str
        Name of the date/time column (will be included in the output).
    k : int
        %K period (default 14).
    d : int
        %D period (default 3).
    mamode : str
        Moving average type for %D (default "sma").
    offset : int
        Shift the result by this many periods.
    fillna : float, optional
        Value to fill NaNs after shifting.
    use_talib : bool
        If True and TA‑Lib is available, use it; otherwise use Numba.
    suffix : str
        Custom suffix for column names (default f"_{k}_{d}").

    Returns
    -------
    pl.DataFrame
        A new DataFrame with columns:
            date_col (as provided),
            STOCHFk{suffix} (fast %K line),
            STOCHFd{suffix} (fast %D line).
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    stoch_k, stoch_d = stochf_numpy(
        high, low, close, k, d, mamode, offset, fillna, use_talib
    )
    suffix = suffix or f"_{k}_{d}"
    return pl.DataFrame({
        date_col: df[date_col],
        f"STOCHFk{suffix}": stoch_k,
        f"STOCHFd{suffix}": stoch_d,
    })