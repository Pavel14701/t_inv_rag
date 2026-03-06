# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# RMA (Wilder's Moving Average) – Numba core
# ----------------------------------------------------------------------
@jit((float64[:], int64), nopython=True, fastmath=True, cache=True)
def _rma_numba_core(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Wilder's Moving Average (RMA) using Numba.
    First value (index length-1) is SMA of first `length` points.
    Then: RMA[i] = RMA[i-1] + (1/length) * (arr[i] - RMA[i-1])
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Initial SMA
    s = 0.0
    for i in range(length):
        s += arr[i]
    out[length - 1] = s / length
    alpha = 1.0 / length
    for i in range(length, n):
        out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return out


# ----------------------------------------------------------------------
# RMA public functions with validation and NaN handling
# ----------------------------------------------------------------------
def rma_numba(
    arr: np.ndarray,
    length: int,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',   # 'raise', 'ffill', 'bfill', 'both'
) -> np.ndarray:
    """
    RMA using Numba with offset, fillna, and NaN handling.

    Parameters
    ----------
    arr : np.ndarray
        Input array (float).
    length : int
        RMA period (>= 1).
    offset : int
        Shift the result by this many periods.
    fillna : float, optional
        Replace NaNs in the result with this value.
    nan_policy : str, default 'raise'
        How to handle NaNs in the input:
        - 'raise': raise ValueError if any NaN is present.
        - 'ffill': forward fill (propagate last valid observation).
        - 'bfill': backward fill (propagate next valid observation).
        - 'both': first forward fill, then backward fill (fills all gaps).

    Returns
    -------
    np.ndarray
        RMA values.
    """
    # ---- Input validation ----
    if length < 1:
        raise ValueError("RMA length must be >= 1")
    arr = np.asarray(arr, dtype=np.float64, copy=False)
    # ---- NaN handling on input ----
    if np.isnan(arr).any():
        if nan_policy == 'raise':
            raise ValueError("Input contains NaN values. \
                Use nan_policy='ffill', 'bfill' or 'both' to fill them.")
        elif nan_policy == 'ffill':
            arr = arr.copy()
            for i in range(1, len(arr)):
                if np.isnan(arr[i]):
                    arr[i] = arr[i - 1]
        elif nan_policy == 'bfill':
            arr = arr.copy()
            for i in range(len(arr) - 2, -1, -1):
                if np.isnan(arr[i]):
                    arr[i] = arr[i + 1]
        elif nan_policy == 'both':
            arr = arr.copy()
            # forward fill
            for i in range(1, len(arr)):
                if np.isnan(arr[i]):
                    arr[i] = arr[i - 1]
            # backward fill (to handle leading NaNs)
            for i in range(len(arr) - 2, -1, -1):
                if np.isnan(arr[i]):
                    arr[i] = arr[i + 1]
        else:
            raise ValueError(f"Unknown nan_policy: {nan_policy}. \
                Use 'raise', 'ffill', 'bfill', or 'both'.")
    # Ensure C-contiguous for Numba performance
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    result = _rma_numba_core(arr, length)
    return _apply_offset_fillna(result, offset, fillna)


def rma_ind(
    arr: np.ndarray | pl.Series,
    length: int,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """
    Universal RMA (always uses Numba) with NaN handling.
    """
    if isinstance(arr, pl.Series):
        arr = arr.to_numpy()
    return rma_numba(arr, length, offset, fillna, nan_policy)


def rma_polars(
    df: pl.DataFrame,
    col: str,
    length: int,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    output_col: str | None = None
) -> pl.DataFrame:
    """
    RMA for Polars DataFrame with NaN handling.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    col : str
        Column name to compute RMA on.
    length : int
        RMA period.
    offset, fillna, nan_policy : as in rma_numba.
    output_col : str, optional
        Name of the output column (default "RMA_{length}").

    Returns
    -------
    pl.DataFrame
        DataFrame with added RMA column.
    """
    arr = df[col].to_numpy()
    result = rma_ind(
        arr, length=length, offset=offset, fillna=fillna, nan_policy=nan_policy
    )
    out_name = output_col or f"RMA_{length}"
    return df.with_columns(pl.Series(out_name, result))