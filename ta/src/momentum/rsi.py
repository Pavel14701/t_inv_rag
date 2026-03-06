# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, njit

from .. import talib, talib_available
from ..overlap import rma_ind
from ..utils import _apply_offset_fillna, _handle_nan_policy


# ----------------------------------------------------------------------
# Numba-accelerated gain/loss calculation
# ----------------------------------------------------------------------
@njit((float64[:], int64), fastmath=True, cache=True)
def _compute_gain_loss_numba(
    close: np.ndarray, 
    drift: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gain and loss arrays from close prices with given drift.
    Returns (gain, loss).
    """
    n = len(close)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(drift, n):
        diff = close[i] - close[i - drift]
        if diff > 0:
            gain[i] = diff
        elif diff < 0:
            loss[i] = -diff
        # else both zero
    return gain, loss


# ----------------------------------------------------------------------
# Numpy‑based RSI calculation with enhancements
# ----------------------------------------------------------------------
def rsi_numpy(
    close: np.ndarray,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """
    Numpy‑based RSI calculation with NaN handling and trim option.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        RSI period.
    scalar : float
        Scaling factor (typically 100).
    drift : int
        Lookback period for price differences.
    offset, fillna, use_talib : as usual.
    nan_policy : str, default 'raise'
        How to handle NaNs in input: 'raise', 'ffill', 'bfill', 'both'.
    trim : bool, default False
        If True, return only the valid part (first `length` values removed).

    Returns
    -------
    np.ndarray
        RSI values.
    """
    # ---- Input validation ----
    if length < 1:
        raise ValueError("RSI length must be >= 1")
    if drift < 1:
        raise ValueError("drift must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    # Check for infinite values
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")
    # Apply NaN policy
    close = _handle_nan_policy(close, nan_policy, "close")
    # Ensure C-contiguous
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # ---- Calculate RSI ----
    if use_talib and talib_available:
        rsi = talib.RSI(close, timeperiod=length)
    else:
        # Use Numba-accelerated gain/loss
        gain, loss = _compute_gain_loss_numba(close, drift)
        avg_gain = rma_ind(gain, length, offset=0, fillna=None, nan_policy=nan_policy)
        avg_loss = rma_ind(loss, length, offset=0, fillna=None, nan_policy=nan_policy)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = scalar - scalar / (1.0 + rs)
            rsi = np.where(avg_loss == 0, scalar, rsi)  # if avg_loss=0, RSI=100
            rsi = np.where(avg_gain == 0, 0.0, rsi)     # if avg_gain=0, RSI=0
            rsi = np.where((avg_gain == 0) & (avg_loss == 0), np.nan, rsi)  # undefined
    # ---- Trim if requested ----
    if trim:
        if len(rsi) >= length:
            rsi = rsi[length - 1:]
        else:
            rsi = np.array([])
    # ---- Apply offset and fillna ----
    return _apply_offset_fillna(rsi, offset, fillna)


# ----------------------------------------------------------------------
# Universal RSI (accepts numpy array or Polars Series)
# ----------------------------------------------------------------------
def rsi_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """
    Universal RSI (accepts numpy array or Polars Series) with NaN handling and trim.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return rsi_numpy(
        close,
        length=length,
        scalar=scalar,
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
def rsi_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add RSI column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Column with close prices.
    length, scalar, drift, offset, fillna, use_talib, nan_policy : as above.
    output_col : str, optional
        Output column name (default f"RSI_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added RSI column.
    """
    close = df[close_col].to_numpy()
    result = rsi_numpy(
        close,
        length=length,
        scalar=scalar,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars always returns full length
    )
    out_name = output_col or f"RSI_{length}"
    return df.with_columns(pl.Series(out_name, result))