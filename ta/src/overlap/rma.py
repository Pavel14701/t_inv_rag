# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, jit

from .._array_ops import _apply_offset_fillna, _handle_nan_policy


# ----------------------------------------------------------------------
# RMA (Wilder's Moving Average) – Numba core
# ----------------------------------------------------------------------
@jit((float64[:], int64), nopython=True, cache=True)
def _rma_numba_core(arr: np.ndarray, length: int) -> np.ndarray:
    """Wilder's Moving Average (RMA) using Numba.
    First value (index length-1) is SMA of first `length` points.
    Then: RMA[i] = RMA[i-1] + (1/length) * (arr[i] - RMA[i-1]).
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
    """RMA using Numba with offset, fillna, and NaN handling.

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
        - 'ignore': leave NaNs as-is (they poison the recursion onward).
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
        raise ValueError('RMA length must be >= 1')
    arr = np.asarray(arr, dtype=np.float64)
    # ---- NaN handling on input ----
    # Supports 'raise', 'ignore', 'ffill', 'bfill', 'both' (same
    # convention as the rest of the codebase).
    arr = _handle_nan_policy(arr, nan_policy, 'input')
    # Numba's typed dispatch requires a writable, C-contiguous array
    # (Polars `.to_numpy()` and read-only views may be non-writable).
    if not (arr.flags.c_contiguous and arr.flags.writeable):
        arr = np.array(arr, dtype=np.float64, copy=True)
    result = _rma_numba_core(arr, length)
    return _apply_offset_fillna(result, offset, fillna)


def rma_ind(
    arr: np.ndarray | pl.Series,
    length: int,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    use_talib: bool = True,
) -> np.ndarray:
    """Universal RMA (always uses Numba) with NaN handling.

    ``use_talib`` is accepted (and ignored) so that ``ma_mode`` can be
    called with a uniform keyword set across MA modes; RMA has no
    TA-Lib backend and always uses the Numba kernel.
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
    """RMA for Polars DataFrame with NaN handling.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    col : str
        Column name to compute RMA on.
    length : int
        RMA period.
    offset : int
        Shift of the output series by ``offset`` bars (default 0).
    fillna : float, optional
        Value used to fill NaNs instead of the default NaN policy.
    nan_policy : str, default 'raise'
        How to handle NaNs in the input ('raise', 'ffill', 'bfill', 'both').
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
    out_name = output_col or f'RMA_{length}'
    return df.with_columns(pl.Series(out_name, result))