# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit, prange

from .smma import _smma_numba_core


# ----------------------------------------------------------------------
# Parallel Alligator core (three lines computed simultaneously)
# ----------------------------------------------------------------------
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _alligator_numba_parallel(
    close: np.ndarray,
    jaw_len: int,
    teeth_len: int,
    lips_len: int,
    offset: int,
    fillna: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the three Alligator lines in parallel using prange.
    Offset and fillna are applied inside.
    """
    n = len(close)
    jaw = np.full(n, np.nan, dtype=np.float64)
    teeth = np.full(n, np.nan, dtype=np.float64)
    lips = np.full(n, np.nan, dtype=np.float64)

    # Parallel loop over the three lines
    for line_idx in prange(3):
        if line_idx == 0:
            length = jaw_len
            out = jaw
        elif line_idx == 1:
            length = teeth_len
            out = teeth
        else:
            length = lips_len
            out = lips
        if n < length:
            continue
        # SMMA calculation for this line
        s = 0.0
        for i in range(length):
            s += close[i]
        out[length - 1] = s / length
        for i in range(length, n):
            out[i] = ((length - 1) * out[i - 1] + close[i]) / length
    # Apply offset (shift)
    if offset != 0:
        for arr in (jaw, teeth, lips):
            if offset > 0:
                arr[offset:] = arr[:-offset]
                arr[:offset] = np.nan
            else:  # offset < 0
                arr[:offset] = arr[-offset:]
                arr[offset:] = np.nan
    # Fill NaNs if requested
    if fillna is not None:
        for arr in (jaw, teeth, lips):
            for i in range(n):
                if np.isnan(arr[i]):
                    arr[i] = fillna

    return jaw, teeth, lips


# ----------------------------------------------------------------------
# Main Alligator function with mode selection
# ----------------------------------------------------------------------
def alligator_ind(
    close: np.ndarray | pl.Series,
    jaw: int = 13,
    teeth: int = 8,
    lips: int = 5,
    offset: int = 0,
    fillna: float | None = None,
    parallel: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bill Williams Alligator indicator.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    jaw, teeth, lips : int
        Periods for each line.
    offset : int
        Shift the result.
    fillna : float, optional
        Value to replace NaNs.
    parallel : bool, default True
        If True, use parallel computation (faster for large data).
        If False, use sequential computation (less overhead for small data).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Jaw, teeth, lips as numpy arrays.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if parallel:
        return _alligator_numba_parallel(close, jaw, teeth, lips, offset, fillna)
    # Sequential mode: compute each line separately
    jaw_arr = _smma_numba_core(close, jaw)
    teeth_arr = _smma_numba_core(close, teeth)
    lips_arr = _smma_numba_core(close, lips)
    # Apply offset and fillna (same logic as in parallel version)
    if offset != 0:
        for arr in (jaw_arr, teeth_arr, lips_arr):
            if offset > 0:
                arr[offset:] = arr[:-offset]
                arr[:offset] = np.nan
            else:
                arr[:offset] = arr[-offset:]
                arr[offset:] = np.nan
    if fillna is not None:
        n = len(close)
        for arr in (jaw_arr, teeth_arr, lips_arr):
            for i in range(n):
                if np.isnan(arr[i]):
                    arr[i] = fillna
    return jaw_arr, teeth_arr, lips_arr


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def alligator_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    jaw: int = 13,
    teeth: int = 8,
    lips: int = 5,
    offset: int = 0,
    fillna: float | None = None,
    parallel: bool = True,
    suffix: str = ""
) -> pl.DataFrame:
    """
    Add Alligator columns (jaw, teeth, lips) to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    jaw, teeth, lips : int
        Periods for each line.
    offset : int
        Global shift for all lines.
    fillna : float, optional
        Value to fill NaNs.
    parallel : bool, default True
        If True, use parallel computation.
    suffix : str
        Suffix appended to column names (default: f"_{jaw}_{teeth}_{lips}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with three new columns.
    """
    close = df[close_col].to_numpy()
    jaw_arr, teeth_arr, lips_arr = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips,
        offset=offset, fillna=fillna, parallel=parallel
    )
    suffix = suffix or f"_{jaw}_{teeth}_{lips}"
    return df.with_columns([
        pl.Series(f"AGj{suffix}", jaw_arr),
        pl.Series(f"AGt{suffix}", teeth_arr),
        pl.Series(f"AGl{suffix}", lips_arr)
    ])