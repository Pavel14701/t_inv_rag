# -*- coding: utf-8 -*-
"""McGinley Dynamic (MCGD) moving average.

This module provides:
- Numba-accelerated core (`_mcgd_numba_core`)
- Numba implementation (`mcgd_numba`) with NaN policy support
- Universal wrapper (`mcgd_ind`) – TA-Lib has no MCGD, so the Numba
  backend is always used
- Polars integration (`mcgd_polars`)

All floating-point operations follow IEEE 754 rules (no fastmath
optimisations). Infinite values are replaced with NaN before calculation.
A NaN in the input poisons the recursive filter from that point onward
(consistent with other recursive moving averages such as EMA).
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
# Core MCGD calculation in Numba (single pass)
# ----------------------------------------------------------------------
@jit(
    'float64[:](float64[:], int64, float64)',
    nopython=True,
    cache=True,
    fastmath=False,
)
def _mcgd_numba_core(close: np.ndarray, length: int, c: float) -> np.ndarray:
    """McGinley Dynamic core loop.

    Formula: MCGD[i] = MCGD[i-1] + (close[i] - MCGD[i-1]) /
             (c * length * (close[i] / MCGD[i-1])**4)

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64), must contain at least one element.
    length : int
        Period parameter (n in the formula), must be >= 1.
    c : float
        Denominator multiplier (usually 1, sometimes 0.6), must be > 0.

    Returns
    -------
    np.ndarray
        MCGD values; the first element equals the first close price.

    Notes
    -----
    - IEEE 754 compliant: NaN propagates naturally through the recursion
        (a single NaN poisons every subsequent value).
    - Degenerate cases where the denominator would be zero (zero price or
        zero previous MCGD) do not raise: the filter carries the previous
        value forward (or re-seeds from the price when MCGD is zero) so the
        series never explodes to +/-Inf.

    """
    n = len(close)
    mcgd = np.empty(n, dtype=np.float64)
    if n == 0:
        return mcgd
    mcgd[0] = close[0]

    for i in range(1, n):
        prev = mcgd[i - 1]
        # Re-seed if the recursion degenerated to zero (denominator would
        # be infinite and the filter would be stuck at zero forever).
        if prev == 0.0:
            mcgd[i] = close[i]
            continue
        ratio = close[i] / prev
        denom = c * length * (ratio ** 4)
        # Guard against division by zero (close[i] == 0 -> ratio == 0).
        if denom == 0.0:
            mcgd[i] = prev
        else:
            mcgd[i] = prev + (close[i] - prev) / denom
    return mcgd


# ----------------------------------------------------------------------
# Public Numba function
# ----------------------------------------------------------------------
def mcgd_numba(
    close: np.ndarray,
    length: int = 10,
    c: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """McGinley Dynamic using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int, default 10
        Period parameter (must be >= 1).
    c : float, default 1.0
        Denominator multiplier (must be > 0).
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaNs in input:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        MCGD values.

    Raises
    ------
    ValueError
        If length < 1, c <= 0, input series is empty, nan_policy is
        invalid, or input contains NaN with nan_policy='raise'.

    Notes
    -----
    - Infinite values are replaced with NaN before any calculation.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError(f'MCGD length must be >= 1, got {length}.')
    if c <= 0:
        raise ValueError(f'MCGD c must be > 0, got {c}.')
    close = np.asarray(close, dtype=np.float64)
    if len(close) < 1:
        raise ValueError(
            f'Input series too short: need at least 1 element, '
            f'got {len(close)}.'
        )

    # Replace infinities with NaN
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, 'close')

    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    mcgd = _mcgd_numba_core(close, length, c)
    return _apply_offset_fillna(mcgd, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def mcgd_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    c: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> np.ndarray:
    """Universal McGinley Dynamic.

    TA-Lib does not provide MCGD, so the Numba backend is always used.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 10
        Period parameter (must be >= 1).
    c : float, default 1.0
        Denominator multiplier (must be > 0).
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaNs in input:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        MCGD values.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return mcgd_numba(close, length, c, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def mcgd_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 10,
    c: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    output_col: str | None = None
) -> pl.DataFrame:
    """Add MCGD column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str, default 'close'
        Column with close prices.
    length : int, default 10
        Period parameter (must be >= 1).
    c : float, default 1.0
        Denominator multiplier (must be > 0).
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        NaN handling policy.
    output_col : str or None, default None
        Output column name (default f"MCGD_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with the MCGD column added.

    Notes
    -----
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = mcgd_ind(
        close,
        length=length,
        c=c,
        offset=offset,
        fillna=fillna,
        nan_policy=nan_policy,
    )
    out_name = output_col or f'MCGD_{length}'
    return df.with_columns([pl.Series(out_name, result)])
