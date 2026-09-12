# -*- coding: utf-8 -*-
"""Alligator indicator (Bill Williams) implementation.

The Alligator consists of three smoothed moving averages (SMMA):
- Jaw (blue line) - period 13, shifted 8 bars forward
- Teeth (red line)  - period 8,  shifted 5 bars forward
- Lips (green line) - period 5,  shifted 3 bars forward

The shift (offset) is applied globally to all lines.
"""
import numpy as np
import polars as pl

from numba import jit, prange

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from .smma import _smma_numba_core


# ----------------------------------------------------------------------
# Parallel Alligator core (three lines computed simultaneously)
# ----------------------------------------------------------------------
@jit(nopython=True, parallel=True, fastmath=False, cache=True)
def _alligator_numba_parallel(
    close: np.ndarray,
    jaw_len: int,
    teeth_len: int,
    lips_len: int,
    offset: int,
    fillna: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three Alligator lines in parallel using prange.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of closing prices.
    jaw_len, teeth_len, lips_len : int
        Periods for each line.
    offset : int
        Shift (positive = forward, negative = backward).
    fillna : float or None
        Value to replace NaNs, or None to keep them.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (jaw, teeth, lips) - all arrays have the same length as `close`.

    Notes
    -----
    - The SMMA is calculated as: first value = SMA of first `period` elements,
        subsequent values = ((period-1)*prev + current) / period.
    - Offset is applied after calculation (shifted positions become NaN unless
        fillna is provided, in which case they become fillna).
    - This function is IEEE 754 compliant (no fastmath optimisations).

    """
    n = len(close)
    jaw = np.full(n, np.nan, dtype=np.float64)
    teeth = np.full(n, np.nan, dtype=np.float64)
    lips = np.full(n, np.nan, dtype=np.float64)
    # Parallel loop over the three lines
    for line_idx in prange(3):  # type: ignore[attr-defined]
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
        # SMMA calculation
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
    parallel: bool = True,
    nan_policy: str = 'raise',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bill Williams Alligator indicator.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices (float64).
    jaw, teeth, lips : int, default 13, 8, 5
        Periods for each line.
    offset : int, default 0
        Global shift for all lines (positive = forward, negative = backward).
    fillna : float or None, default None
        Value to replace NaNs (including shifted-in positions). If None, NaNs
        remain.
    parallel : bool, default True
        If True, use parallel computation (faster for large data).
        If False, use sequential computation (less overhead for small data).
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (jaw, teeth, lips) as numpy arrays, same length as `close`.

    Raises
    ------
    ValueError
        If any period is < 1, the input contains NaN with
        `nan_policy='raise'`, or `nan_policy` is unknown.

    Notes
    -----
    - The SMMA is computed using the same logic as in the `smma` module.
    - Infinites in `close` are replaced with NaN to avoid propagation of inf.
    - All operations are IEEE 754 compliant.

    """
    if jaw < 1 or teeth < 1 or lips < 1:
        raise ValueError('jaw, teeth and lips periods must all be >= 1')
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Replace infinities with NaN (IEEE 754 compliance)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if parallel:
        return _alligator_numba_parallel(
            close, jaw, teeth, lips, offset, fillna
        )
    # Sequential mode: compute each line separately using _smma_numba_core
    jaw_arr = _smma_numba_core(close, jaw)
    teeth_arr = _smma_numba_core(close, teeth)
    lips_arr = _smma_numba_core(close, lips)
    # Apply offset and fillna (same logic as in parallel version)
    jaw_arr = _apply_offset_fillna(jaw_arr, offset, fillna)
    teeth_arr = _apply_offset_fillna(teeth_arr, offset, fillna)
    lips_arr = _apply_offset_fillna(lips_arr, offset, fillna)
    return jaw_arr, teeth_arr, lips_arr


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def alligator_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    jaw: int = 13,
    teeth: int = 8,
    lips: int = 5,
    offset: int = 0,
    fillna: float | None = None,
    parallel: bool = True,
    suffix: str = '',
) -> pl.DataFrame:
    """Add Alligator columns (jaw, teeth, lips) to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column with close prices.
    jaw, teeth, lips : int, default 13, 8, 5
        Periods for each line.
    offset : int, default 0
        Global shift for all lines.
    fillna : float or None, default None
        Value to replace NaNs.
    parallel : bool, default True
        If True, use parallel computation.
    suffix : str, default ''
        Custom suffix for column names. If empty, a default suffix
        f'_{jaw}_{teeth}_{lips}' is used.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with three new columns:
        - AGj{suffix} (jaw)
        - AGt{suffix} (teeth)
        - AGl{suffix} (lips)

    """
    close = df[close_col].to_numpy()
    jaw_arr, teeth_arr, lips_arr = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips,
        offset=offset, fillna=fillna, parallel=parallel
    )

    if not suffix:
        suffix = f'_{jaw}_{teeth}_{lips}'

    return df.with_columns([
        pl.Series(f'AGj{suffix}', jaw_arr),
        pl.Series(f'AGt{suffix}', teeth_arr),
        pl.Series(f'AGl{suffix}', lips_arr),
    ])
