# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Numba‑accelerated rolling sums of positive and negative changes
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _cmo_numba_core(
    close: np.ndarray,
    length: int,
    drift: int,
    scalar: float,
) -> np.ndarray:
    """
    Compute CMO using sliding window sums of positive and negative changes.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.
    drift : int
        Shift for price differences.
    scalar : float
        Multiplier (e.g., 100.0).

    Returns
    -------
    np.ndarray
        CMO values; first (length+drift-1) positions are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length + drift:
        return out
    # Pre‑compute differences (first `drift` values remain NaN)
    diff = np.empty(n, dtype=np.float64)
    for i in range(drift, n):
        diff[i] = close[i] - close[i - drift]
    # Cumulative sums for positive and negative parts
    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)
    # Rolling sums using cumulative sums
    cum_pos = np.zeros(n + 1, dtype=np.float64)
    cum_neg = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        cum_pos[i] = cum_pos[i - 1] + pos[i - 1]
        cum_neg[i] = cum_neg[i - 1] + neg[i - 1]
    for i in range(length + drift - 1, n):
        sum_pos = cum_pos[i + 1] - cum_pos[i + 1 - length]
        sum_neg = cum_neg[i + 1] - cum_neg[i + 1 - length]
        denom = sum_pos + sum_neg
        if denom != 0.0:
            out[i] = scalar * (sum_pos - sum_neg) / denom
        else:
            out[i] = np.nan
    return out


def cmo_numpy(
    close: np.ndarray,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based CMO calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.
    scalar : float
        Multiplier.
    drift : int
        Shift for price differences.
    offset, fillna : as usual.
    use_talib : bool
        If True and TA‑Lib is available, use talib.CMO; else use Numba core.

    Returns
    -------
    np.ndarray
        CMO values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if use_talib and talib_available:
        # TA‑Lib CMO uses RMA internally; scalar is fixed at 100.
        result = talib.CMO(close, timeperiod=length)
    else:
        result = _cmo_numba_core(close, length, drift, scalar)
    return _apply_offset_fillna(result, offset, fillna)


def cmo_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal CMO (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return cmo_numpy(close, length, scalar, drift, offset, fillna, use_talib)


def cmo_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add CMO column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, scalar, drift, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"CMO_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    result = cmo_numpy(close, length, scalar, drift, offset, fillna, use_talib)
    out_name = output_col or f"CMO_{length}"
    return df.with_columns([pl.Series(out_name, result)])