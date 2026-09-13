# -*- coding: utf-8 -*-
"""T3 Moving Average (Tim Tillson) - Numba-accelerated with TA-Lib fallback.

T3 is a six-fold EMA combination: T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3.

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation. Leading NaNs are skipped (like TA-Lib
does); a NaN inside the series poisons the tail of the output.
"""

from typing import Optional

import numpy as np
import polars as pl

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available
from ..overlap import ema_ind


def _first_valid_index(arr: np.ndarray) -> int:
    """Return the index of the first finite value, or -1 if there is none."""
    idx = np.flatnonzero(np.isfinite(arr))
    return int(idx[0]) if idx.size else -1


# ----------------------------------------------------------------------
# Core T3 calculation using Numba (six-fold EMA)
# ----------------------------------------------------------------------
def t3_numba(
    close: np.ndarray,
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None,
    nan_policy: str = "raise",
) -> np.ndarray:
    """T3 moving average using Numba (six-fold EMA).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int, default 10
        EMA period.
    a : float, default 0.7
        Volume factor.
    offset : int, default 0
        Shift the result.
    fillna : float or None, default None
        Value to fill NaNs.
    nan_policy : str, default 'raise'
        How to handle NaN values: 'raise', 'ignore', 'ffill', 'bfill', 'both'.

    Returns
    -------
    np.ndarray
        T3 values; the first ``6 * (length - 1)`` positions are NaN
        (same lookback as TA-Lib).

    Raises
    ------
    ValueError
        If `length < 1` or `nan_policy == 'raise'` and NaNs are present.

    Notes
    -----
    - Infinites in `close` are replaced with NaN before calculation.
    - Leading NaNs are skipped: each EMA stage seeds on the first valid
      values (like TA-Lib). A NaN inside the series poisons the tail.
    - Empty or too-short inputs return an all-NaN array.
    - This function is IEEE 754 compliant.

    """
    if length < 1:
        raise ValueError("T3 length must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return _apply_offset_fillna(out, offset, fillna)

    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")

    # Lookback of the six-fold EMA is 6 * (length - 1); a shorter series
    # cannot produce a single value (TA-Lib returns all-NaN as well).
    if n <= 6 * (length - 1):
        return _apply_offset_fillna(out, offset, fillna)

    k = _first_valid_index(close)
    if k < 0:
        return _apply_offset_fillna(out, offset, fillna)

    # Tillson coefficients
    a2 = a * a
    a3 = a2 * a
    c1 = -a3
    c2 = 3.0 * a2 + 3.0 * a3
    c3 = -6.0 * a2 - 3.0 * a - 3.0 * a3
    c4 = 1.0 + 3.0 * a + 3.0 * a2 + a3

    # Six successive EMAs. Each EMA seeds on the first `length` values of
    # its input, so a NaN warm-up head would poison the seed. The head is
    # therefore sliced off between stages (its size is tracked in `start`).
    start = k
    tails: list[tuple[int, np.ndarray]] = []
    work = close[k:]
    for _ in range(6):
        em = ema_ind(work, length=length, use_talib=False, nan_policy="ignore")
        m = _first_valid_index(em)
        if m < 0:
            return _apply_offset_fillna(out, offset, fillna)
        work = em[m:]
        start += m
        tails.append((start, work))

    _, e3 = tails[2]
    _, e4 = tails[3]
    _, e5 = tails[4]
    _, e6 = tails[5]

    # All tails end at the last bar; align them by their common suffix.
    L = len(e6)  # noqa: N806 - formula symbol
    if L <= 0:
        return _apply_offset_fillna(out, offset, fillna)
    t3 = c1 * e6 + c2 * e5[-L:] + c3 * e4[-L:] + c4 * e3[-L:]
    out[n - L :] = t3
    return _apply_offset_fillna(out, offset, fillna)


# ----------------------------------------------------------------------
# TA-Lib wrapper
# ----------------------------------------------------------------------
def t3_talib(
    close: np.ndarray,
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> np.ndarray:
    """T3 using TA-Lib (C implementation).

    Notes
    -----
    - Infinites in `close` are replaced with NaN before calculation
      (IEEE 754 compliance).

    """
    if length < 1:
        raise ValueError("T3 length must be >= 1")
    if not talib_available:
        raise ImportError("TA-Lib not available")
    close = np.asarray(close, dtype=np.float64).copy()
    replace_inf_with_nan(close)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    t3 = talib.T3(close, timeperiod=length, vfactor=a)
    return _apply_offset_fillna(t3, offset, fillna)


# ----------------------------------------------------------------------
# Universal wrapper
# ----------------------------------------------------------------------
def t3_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal T3 moving average.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        EMA period.
    a : float
        Volume factor (0 < a < 1).
    offset, fillna : as usual.
    use_talib : bool
        Use TA-Lib if available.
    nan_policy : str, default 'raise'
        How to handle NaN values (Numba backend only).

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    np.ndarray
        T3 values.

    Notes
    -----
    - All operations are IEEE 754 compliant.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return t3_talib(close, length, a, offset, fillna)
    else:
        return t3_numba(close, length, a, offset, fillna, nan_policy)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def t3_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    a: float = 0.7,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: Optional[str] = None,
) -> pl.DataFrame:
    """Add T3 column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, a, offset, fillna, use_talib, nan_policy : as above.
    output_col : str, optional
        Output column name (default f"T3_{length}_{a}").

    a : float, optional
        See the module guide; default mirrors the numpy path.
    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    length : int, optional
        See the module guide; default mirrors the numpy path.
    nan_policy : str, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.
    use_talib : bool, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with T3 column.

    """
    close = df[close_col].to_numpy()
    result = t3_ind(close, length, a, offset, fillna, use_talib, nan_policy)
    out_name = output_col or f"T3_{length}_{a}"
    return df.with_columns([pl.Series(out_name, result)])
