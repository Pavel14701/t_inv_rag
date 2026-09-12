# -*- coding: utf-8 -*-
"""Holt-Winter Moving Average (HWMA) implementation.

This module provides:
- Numba-accelerated core (`_hwma_numba_core`)
- Numba wrapper with NaN handling, offset/fillna (`hwma_numba`)
- Universal wrapper (`hwma_ind`)
- Polars integration (`hwma_polars`)

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation.
"""

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)


# ----------------------------------------------------------------------
# Core HWMA loop in Numba
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True, fastmath=False)
def _hwma_numba_core(
    close: np.ndarray, na: float, nb: float, nc: float
) -> np.ndarray:
    """Holt-Winter Moving Average core loop.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of prices (assumed to have no NaNs or infinities).
    na, nb, nc : float
        Smoothing parameters (0 < parameter < 1).

    Returns
    -------
    np.ndarray
        HWMA values (same length as close).

    Notes
    -----
    - This function assumes `close` has no NaNs or infinities.
    - NaN/Inf handling is done in the caller.

    """
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    last_a = 0.0
    last_v = 0.0
    last_f = close[0]
    for i in range(n):
        f = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close[i]
        v = (1.0 - nb) * (last_v + last_a) + nb * (f - last_f)
        a = (1.0 - nc) * last_a + nc * (v - last_v)
        out[i] = f + v + 0.5 * a
        last_a, last_f, last_v = a, f, v
    return out


# ----------------------------------------------------------------------
# HWMA using Numba (with NaN handling, offset and fillna)
# ----------------------------------------------------------------------
def hwma_numba(
    close: np.ndarray,
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Holt-Winter Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    na, nb, nc : float
        Smoothing parameters (must be strictly between 0 and 1).
    offset : int, default 0
        Shift result. Positive = forward, negative = backward.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        HWMA values, same length as `close`.

    Raises
    ------
    ValueError
        If `na`, `nb` or `nc` is not in the open interval (0, 1), the input
        contains NaN with `nan_policy='raise'`, or `nan_policy` is unknown.

    Notes
    -----
    - Infinites in `close` are replaced with NaN before calculation.
    - This function is IEEE 754 compliant.

    """
    if not (0.0 < na < 1.0):
        raise ValueError(f"na must be in (0, 1), got {na}")
    if not (0.0 < nb < 1.0):
        raise ValueError(f"nb must be in (0, 1), got {nb}")
    if not (0.0 < nc < 1.0):
        raise ValueError(f"nc must be in (0, 1), got {nc}")
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Replace infinities with NaN (IEEE 754 compliance)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    hwma = _hwma_numba_core(close, na, nb, nc)
    return _apply_offset_fillna(hwma, offset, fillna)


# ----------------------------------------------------------------------
# Universal HWMA function
# ----------------------------------------------------------------------
def hwma_ind(
    close: np.ndarray | pl.Series,
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal Holt-Winter Moving Average (always uses Numba).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    na, nb, nc : float
        Smoothing parameters (0 < param < 1).
    offset : int, default 0
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in `close`.

    Returns
    -------
    np.ndarray
        HWMA values.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return hwma_numba(close, na, nb, nc, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def hwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """HWMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column with close prices.
    na, nb, nc : float
        Smoothing parameters (0 < param < 1).
    offset : int, default 0
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close column.
    output_col : str, optional
        Output column name (default f"HWMA_{na}_{nb}_{nc}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with the added HWMA column.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = hwma_ind(
        close,
        na=na,
        nb=nb,
        nc=nc,
        offset=offset,
        fillna=fillna,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"HWMA_{na}_{nb}_{nc}"
    return df.with_columns([pl.Series(out_name, result)])
