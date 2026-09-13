# -*- coding: utf-8 -*-
"""VIDYA (Variable Index Dynamic Average) implementation.

VIDYA is an adaptive EMA whose smoothing constant is scaled by the absolute
Chande Momentum Oscillator (CMO) of the input series.

This module provides:
- Numba-accelerated CMO core (`_cmo_numba`)
- Numba implementation (`vidya_numba`) with NaN policy support
- TA-Lib based implementation (`vidya_talib`) using TA-Lib CMO
- Universal wrapper (`vidya_ind`)
- Polars integration (`vidya_polars`)

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
from ..external import talib, talib_available


# ----------------------------------------------------------------------
# CMO (Chande Momentum Oscillator) - Numba version
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True, fastmath=False)
def _cmo_numba(close: np.ndarray, length: int, drift: int) -> np.ndarray:
    """Compute Chande Momentum Oscillator (CMO) in [-1, 1] using Numba.

    Returns an array of CMO values, same length as `close`.
    The first (length + drift - 1) values are NaN (undefined window).
    NaN values in `close` propagate to all subsequent CMO values
    (IEEE 754 semantics).
    """
    n = len(close)
    cmo = np.full(n, np.nan, dtype=np.float64)
    # Need at least length+drift to compute anything
    if n < length + drift:
        return cmo
    # We'll compute rolling sums of positive and negative changes.
    # Pre-compute differences. The first `drift` slots are undefined and
    # are zero-initialised so cumulative sums never contain garbage
    # (np.empty would leak uninitialised memory into the results).
    diff = np.zeros(n, dtype=np.float64)
    for i in range(drift, n):
        diff[i] = close[i] - close[i - drift]
    # Rolling sums using cumulative sum trick
    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)
    # Cumulative sums
    cum_pos = np.zeros(n + 1, dtype=np.float64)
    cum_neg = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        cum_pos[i] = cum_pos[i - 1] + pos[i - 1]
        cum_neg[i] = cum_neg[i - 1] + neg[i - 1]
    # Rolling sums over window `length` covering diffs [i-length+1, i].
    # The window is fully defined for i >= length + drift - 1, which for
    # drift=1 matches TA-Lib CMO (first valid output index = length).
    for i in range(length + drift - 1, n):
        pos_sum = cum_pos[i + 1] - cum_pos[i - length + 1]
        neg_sum = cum_neg[i + 1] - cum_neg[i - length + 1]
        denom = pos_sum + neg_sum
        if denom != 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
            cmo[i] = (pos_sum - neg_sum) / denom
        else:
            cmo[i] = 0.0  # avoid division by zero
    return cmo


# ----------------------------------------------------------------------
# VIDYA using Numba
# ----------------------------------------------------------------------
def vidya_numba(
    close: np.ndarray,
    length: int = 14,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """VIDYA using Numba with IEEE 754 compliant NaN/Inf handling.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 14
        Period for CMO and the EMA smoothing constant.
    drift : int, default 1
        Lag for price differences inside the CMO calculation.
    offset : int, default 0
        Shift applied to the result. Positive = forward.
    fillna : float or None, default None
        Value to replace NaN after shift.
    nan_policy : str, default 'raise'
        How to handle NaNs in input:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        VIDYA values. The first `length + drift - 1` elements are NaN
        (undefined CMO window); the recursion is seeded with the price at
        the first valid index.

    Raises
    ------
    ValueError
        If length < 1, drift < 1, input series too short, or invalid
        nan_policy.

    Notes
    -----
    - Infinities are replaced with NaN before calculation.
    - A NaN in the input poisons the recursive filter from that point
      onward (IEEE 754 propagation, consistent with EMA-like filters).
    - This function is IEEE 754 compliant (no fastmath).

    """
    if length < 1:
        raise ValueError(f"VIDYA length must be >= 1, got {length}.")
    if drift < 1:
        raise ValueError(f"VIDYA drift must be >= 1, got {drift}.")
    if nan_policy not in ("raise", "ignore", "ffill", "bfill", "both"):
        raise ValueError(
            f"Unknown nan_policy: {nan_policy}. "
            "Use 'raise', 'ignore', 'ffill', 'bfill', or 'both'."
        )
    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    n = len(close)
    if n < length + drift:
        raise ValueError(
            f"Input series too short: need at least {length + drift} "
            f"elements, got {n}."
        )

    alpha = 2.0 / (length + 1.0)
    cmo = _cmo_numba(close, length, drift)
    abs_cmo = np.abs(cmo)

    vidya = np.full(n, np.nan, dtype=np.float64)
    start = length + drift - 1  # first index with a fully defined CMO window
    if not np.isnan(abs_cmo[start]) and not np.isnan(close[start]):
        # Seed the recursion with the price at the first valid index
        # (seeding with the NaN CMO warmup values would poison the
        # whole output with NaN).
        vidya[start] = close[start]
        for i in range(start + 1, n):
            if np.isnan(abs_cmo[i]) or np.isnan(close[i]):
                break  # NaN poisons the recursive filter from here on
            sc = alpha * abs_cmo[i]
            vidya[i] = sc * close[i] + (1.0 - sc) * vidya[i - 1]
    return _apply_offset_fillna(vidya, offset, fillna)


# ----------------------------------------------------------------------
# VIDYA using TA-Lib (if available)
# ----------------------------------------------------------------------
def vidya_talib(
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """VIDYA using TA-Lib CMO (drift=1) plus a Python recursion.

    TA-Lib provides CMO but not VIDYA directly. TA-Lib's CMO ranges over
    [-100, 100]; it is normalised to [0, 1] here so that the adaptive
    smoothing scale matches `vidya_numba` with ``drift=1``.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 14
        Period for CMO and the EMA smoothing constant.
    offset : int, default 0
        Shift applied to the result.
    fillna : float or None, default None
        Value to replace NaN after shift.

    Returns
    -------
    np.ndarray
        VIDYA values. The first `length` elements are NaN.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.
    ValueError
        If length < 1 or input series too short.

    Notes
    -----
    - Infinities are replaced with NaN before calculation.
    - A NaN in the input poisons the recursive filter from that point
      onward (IEEE 754 propagation).

    """
    if not talib_available:
        raise ImportError("TA-Lib not available")
    if length < 1:
        raise ValueError(f"VIDYA length must be >= 1, got {length}.")
    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    n = len(close)
    if n < length + 1:
        raise ValueError(
            f"Input series too short: need at least {length + 1} "
            f"elements, got {n}."
        )

    alpha = 2.0 / (length + 1.0)
    cmo = talib.CMO(close, timeperiod=length)  # range [-100, 100]
    abs_cmo = np.abs(cmo) * 0.01  # normalise to [0, 1]

    vidya = np.full(n, np.nan, dtype=np.float64)
    start = length  # TA-Lib CMO first valid index (drift = 1)
    if not np.isnan(abs_cmo[start]) and not np.isnan(close[start]):
        vidya[start] = close[start]
        for i in range(start + 1, n):
            if np.isnan(abs_cmo[i]) or np.isnan(close[i]):
                break  # NaN poisons the recursive filter from here on
            sc = alpha * abs_cmo[i]
            vidya[i] = sc * close[i] + (1.0 - sc) * vidya[i - 1]
    return _apply_offset_fillna(vidya, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def vidya_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal VIDYA with automatic backend selection.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, default 14
        Period for CMO and alpha.
    drift : int, default 1
        Shift for price differences (used only in the Numba backend).
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    use_talib : bool, default True
        If True and TA-Lib is available, use TA-Lib for CMO
        (drift is fixed to 1 on this backend).
    nan_policy : str, default 'raise'
        How to handle NaNs in input (only for the Numba backend):
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        VIDYA values.

    Notes
    -----
    - If `close` is a Polars Series, it is converted to NumPy.
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return vidya_talib(close, length, offset, fillna)
    return vidya_numba(close, length, drift, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def vidya_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add VIDYA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str, default 'close'
        Column with close prices.
    length : int, default 14
        Period for CMO and alpha.
    drift : int, default 1
        Shift for price differences (Numba backend only).
    offset : int, default 0
        Shift result.
    fillna : float or None, default None
        Value to fill NaNs.
    use_talib : bool, default True
        Use TA-Lib if available.
    nan_policy : str, default 'raise'
        NaN handling policy (Numba backend only).
    output_col : str or None, default None
        Output column name (default f"VIDYA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with VIDYA column.

    Notes
    -----
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    result = vidya_ind(
        close,
        length=length,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"VIDYA_{length}"
    return df.with_columns([pl.Series(out_name, result)])
