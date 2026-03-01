import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# RMA (Wilder's Moving Average) – Numba core
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
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
# RMA public functions (Numba only, no TA-Lib equivalent)
# ----------------------------------------------------------------------
def rma_numba(
    arr: np.ndarray,
    length: int,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    RMA using Numba with offset and fillna (optimized).
    """
    arr = np.asarray(arr, dtype=np.float64, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    result = _rma_numba_core(arr, length)
    return _apply_offset_fillna(result, offset, fillna)


def rma_ind(
    arr: np.ndarray | pl.Series,
    length: int,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal RMA (always uses Numba).
    """
    if isinstance(arr, pl.Series):
        arr = arr.to_numpy()
    return rma_numba(arr, length, offset, fillna)


def rma_polars(
    df: pl.DataFrame,
    col: str,
    length: int,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.Series:
    """
    RMA for Polars DataFrame.
    """
    arr = df[col].to_numpy()
    result = rma_ind(arr, length=length, offset=offset, fillna=fillna)
    out_name = output_col or f"RMA_{length}"
    return pl.Series(out_name, result)
