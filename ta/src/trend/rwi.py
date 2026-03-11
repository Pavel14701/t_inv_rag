# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, njit

from ..utils import _apply_offset_fillna, _handle_nan_policy
from ..volatility import atr_ind


# ----------------------------------------------------------------------
# Numba-ядро для вычисления RWI
# ----------------------------------------------------------------------
@njit((float64[:], float64[:], float64[:], int64), fastmath=True, cache=True)
def _rwi_numba_core(
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-ускоренное вычисление RWI high и low.

    Parameters
    ----------
    high, low : np.ndarray
        Цены high и low (одинаковой длины, предполагается без NaN).
    atr : np.ndarray
        ATR значения (той же длины).
    length : int
        Период сдвига.

    Returns
    -------
    (rwi_high, rwi_low) как numpy массивы.
    """
    n = len(high)
    rwi_high = np.full(n, np.nan, dtype=np.float64)
    rwi_low = np.full(n, np.nan, dtype=np.float64)
    denom_factor = np.sqrt(length)
    for i in range(length, n):
        denom = atr[i] * denom_factor
        if denom != 0.0:
            # RWI high = (high[i] - low[i - length]) / denom
            rwi_high[i] = (high[i] - low[i - length]) / denom
            # RWI low = (high[i - length] - low[i]) / denom
            rwi_low[i] = (high[i - length] - low[i]) / denom
        # else остаётся NaN
    return rwi_high, rwi_low


# ----------------------------------------------------------------------
# Core RWI calculation (NumPy + Numba)
# ----------------------------------------------------------------------
def rwi_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numpy‑based Random Walk Index (RWI) calculation.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays.
    length : int
        Period for ATR and shift (>= 1).
    mamode : str
        Moving average mode for ATR ('rma', 'sma', 'ema').
    drift : int
        Lookback period for ATR (not used for shift, only for ATR).
    offset, fillna, use_talib, nan_policy, trim : as usual.

    Returns
    -------
    (rwi_high, rwi_low) as numpy arrays.
    """
    # ---- Validation ----
    if length < 1:
        raise ValueError("length must be >= 1")
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    for name, arr in [("high", high), ("low", low), ("close", close)]:
        if np.isinf(arr).any():
            raise ValueError(f"Input {name} contains non-finite values (inf or -inf).")
    high = _handle_nan_policy(high, nan_policy, "high")
    low = _handle_nan_policy(low, nan_policy, "low")
    close = _handle_nan_policy(close, nan_policy, "close")
    if not (
        high.flags.c_contiguous and 
        low.flags.c_contiguous and 
        close.flags.c_contiguous
    ):
        high = np.ascontiguousarray(high)
        low = np.ascontiguousarray(low)
        close = np.ascontiguousarray(close)
    n = len(close)
    min_required = length + 1
    if n < min_required:
        raise ValueError(
            f"Input series too short: need at \
                least {min_required} elements, got {n}."
            )
    # ---- ATR ----
    atr = atr_ind(
        high, low, close,
        length=length,
        mamode=mamode,
        drift=drift,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # ATR returns full length
    )
    if np.all(np.isnan(atr)):
        raise ValueError("ATR calculation failed.")
    # ---- Вычисление RWI через Numba-ядро ----
    rwi_high, rwi_low = _rwi_numba_core(high, low, atr, length)
    # ---- Trim ----
    if trim:
        start = length  # первый валидный индекс
        if start < n:
            rwi_high = rwi_high[start:]
            rwi_low = rwi_low[start:]
        else:
            rwi_high = np.array([])
            rwi_low = np.array([])
    # ---- Offset & fillna ----
    rwi_high = _apply_offset_fillna(rwi_high, offset, fillna)
    rwi_low = _apply_offset_fillna(rwi_low, offset, fillna)
    return rwi_high, rwi_low


# ----------------------------------------------------------------------
# Universal RWI (numpy / polars)
# ----------------------------------------------------------------------
def rwi_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal Random Walk Index (accepts numpy arrays or Polars Series).
    Returns (rwi_high, rwi_low) as numpy arrays.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return rwi_numpy(
        high, low, close,
        length=length,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=trim,
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def rwi_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    length: int = 14,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add RWI columns to Polars DataFrame.

    Columns added:
        RWI_HIGH{suffix}
        RWI_LOW{suffix}

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col, low_col, close_col : str
        Column names for prices.
    length, mamode, drift, offset, fillna, use_talib, nan_policy : as in rwi_numpy.
    suffix : str, optional
        Suffix for output column names (default f"_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added columns (same length).
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    rwi_high, rwi_low = rwi_numpy(
        high, low, close,
        length=length,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars always returns full length
    )
    suffix = suffix or f"_{length}"
    return df.with_columns([
        pl.Series(f"RWI_HIGH{suffix}", rwi_high),
        pl.Series(f"RWI_LOW{suffix}", rwi_low),
    ])
