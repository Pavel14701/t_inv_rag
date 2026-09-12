# -*- coding: utf-8 -*-
"""True Range (TR) implementation.

True Range measures the bar-to-bar volatility range including gaps:
``TR = max(high - low, |high - prev_close|, |low - prev_close|)``
with ``prev_close = close[i - drift]`` (drift >= 1).

The module provides:
- ``_true_range_numba_core`` - Numba core loop
- ``true_range_numba`` - Numba wrapper with NaN handling, offset/fillna
- ``true_range_talib`` - TA-Lib backend (fixed drift=1)
- ``true_range_ind`` - universal wrapper with automatic backend
  selection (drift != 1 forces the Numba backend)
- ``true_range_polars`` - Polars DataFrame integration

All floating-point operations follow IEEE 754 rules (no fastmath).
Infinite values are replaced with NaN before calculation.
"""

import numpy as np
import polars as pl

from numba import njit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available


# ----------------------------------------------------------------------
# True Range - Numba core
# ----------------------------------------------------------------------
@njit(fastmath=False, cache=True)
def _true_range_numba_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int,
) -> np.ndarray:
    """Compute True Range (TR) using Numba.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64), same length, no NaNs or infinities.
    drift : int
        Shift for previous close (>= 1).

    Returns
    -------
    np.ndarray
        TR array; the first ``drift`` values are NaN.

    """
    n = len(high)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n <= drift:
        return tr
    for i in range(drift, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - drift])
        lc = abs(low[i] - close[i - drift])
        # numba max propagates NaN (IEEE semantics): any NaN component
        # yields a NaN True Range
        tr[i] = max(hl, hc, lc)
    return tr


def _validate_tr_inputs(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int,
) -> None:
    """Validate True Range inputs; raise ValueError on inconsistency.

    Args:
        high: High price array.
        low: Low price array.
        close: Close price array.
        drift: Shift for the previous close (must be >= 1).

    Raises:
    ------
    ValueError
        If ``drift < 1``, the arrays have different lengths, or the
        series is too short (need at least ``drift + 1`` elements).

    """
    if drift < 1:
        raise ValueError(
            f"TR drift must be >= 1 (got {drift}); drift < 1 would "
            "either compare a bar with itself or look into the future."
        )
    if not (len(high) == len(low) == len(close)):
        raise ValueError(
            f"high, low and close must have the same length: "
            f"got {len(high)}, {len(low)} and {len(close)}."
        )
    if len(high) < drift + 1:
        raise ValueError(
            f"Input series too short: need at least {drift + 1} "
            f"elements, got {len(high)}."
        )


def true_range_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int = 1,
    prenan: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """True Range using Numba.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays, same length.
    drift : int
        Shift for previous close (must be >= 1).
    prenan : bool
        No-op: the first ``drift`` values are always NaN (kept for
        pandas_ta API compatibility).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in ``high``/``low``/``close``:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    np.ndarray
        TR values; the first ``drift`` positions are NaN (or ``fillna``).

    """
    _validate_tr_inputs(high, low, close, drift)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    # Replace infinities with NaN, then apply the NaN policy
    high = high.copy()
    low = low.copy()
    close = close.copy()
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    replace_inf_with_nan(close)
    high = _handle_nan_policy(high, nan_policy, "high")
    low = _handle_nan_policy(low, nan_policy, "low")
    close = _handle_nan_policy(close, nan_policy, "close")

    tr = _true_range_numba_core(high, low, close, drift)
    return _apply_offset_fillna(tr, offset, fillna)


# ----------------------------------------------------------------------
# True Range - TA-Lib wrapper
# ----------------------------------------------------------------------
def true_range_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int = 1,
    prenan: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """True Range using TA-Lib (C implementation, fixed drift=1).

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays, same length.
    drift : int
        Must be 1: TA-Lib always uses the previous bar's close.  Any
        other value raises ValueError (TA-Lib cannot honour it).
    prenan : bool
        Ignored (TA-Lib returns NaN for the first bar anyway).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values in ``high``/``low``/``close``.

    Returns
    -------
    np.ndarray
        TR values; the first position is NaN (or ``fillna``).

    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    if drift != 1:
        raise ValueError(
            f"TA-Lib TRANGE supports drift=1 only, got drift={drift}. "
            "Use use_talib=False for the Numba backend."
        )
    _validate_tr_inputs(high, low, close, drift)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    # Replace infinities with NaN, then apply the NaN policy
    high = high.copy()
    low = low.copy()
    close = close.copy()
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    replace_inf_with_nan(close)
    high = _handle_nan_policy(high, nan_policy, "high")
    low = _handle_nan_policy(low, nan_policy, "low")
    close = _handle_nan_policy(close, nan_policy, "close")

    tr = talib.TRANGE(high, low, close)
    return _apply_offset_fillna(tr, offset, fillna)


# ----------------------------------------------------------------------
# Universal True Range function
# ----------------------------------------------------------------------
def true_range_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    drift: int = 1,
    prenan: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal True Range with automatic backend selection.

    Backend rules:
    - ``drift != 1`` **forces the Numba backend** (TA-Lib TRANGE is
        hard-wired to drift=1 and would silently return wrong values).
    - Otherwise TA-Lib is used when ``use_talib=True`` and available.

    Parameters
    ----------
    high, low, close : np.ndarray or pl.Series
        Price series, same length.
    drift : int
        Shift for previous close (must be >= 1).
    prenan : bool
        No-op (first ``drift`` values are always NaN).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA-Lib if available (drift=1 only).
    nan_policy : str, default 'raise'
        How to handle NaN values in ``high``/``low``/``close``.

    Returns
    -------
    np.ndarray
        TR values.

    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available and drift == 1:
        return true_range_talib(
            high, low, close, drift, prenan, offset, fillna, nan_policy
        )
    return true_range_numba(
        high, low, close, drift, prenan, offset, fillna, nan_policy
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def true_range_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    drift: int = 1,
    prenan: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """True Range for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col, low_col, close_col : str
        Column names for prices.
    drift : int
        Shift for previous close (must be >= 1; drift != 1 forces the
        Numba backend).
    prenan : bool
        No-op (first ``drift`` values are always NaN).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA-Lib if available (drift=1 only).
    nan_policy : str, default 'raise'
        How to handle NaN values in the price columns.
    output_col : str, optional
        Output column name (default f"TRUERANGE_{drift}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added column.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = true_range_ind(
        high,
        low,
        close,
        drift=drift,
        prenan=prenan,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    out_name = output_col or f"TRUERANGE_{drift}"
    return df.with_columns([pl.Series(out_name, result)])
