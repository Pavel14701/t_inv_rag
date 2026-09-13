# -*- coding: utf-8 -*-
"""Ultimate Oscillator (UO).

Larry Williams' oscillator combining three weighted averages of
buying pressure over true range:

    bp  = close - min(low, prev_close)
    tr  = max(high, prev_close) - min(low, prev_close)
    avgN = sum(bp, N) / sum(tr, N)
    UO = 100 * (4*avg_fast + 2*avg_medium + avg_slow) / 7

Defaults: fast=7, medium=14, slow=28. TA-Lib (ULTOSC) is used when
available (semantics match exactly); otherwise a Numba kernel runs.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates: a window touching a NaN bar yields NaN UO;
- a zero total true range in a window yields NaN for that average
  (0/0 undefined), which propagates to the final value.
"""

import numpy as np
import polars as pl

from numba import float64, int64, njit

from .._array_ops import _apply_offset_fillna
from ..external import talib, talib_available


@njit(
    (float64[:], float64[:], float64[:], int64, int64, int64),
    fastmath=False,
    cache=True,
)
def _uo_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fast: int,
    medium: int,
    slow: int,
) -> np.ndarray:
    """Ultimate Oscillator core (fastmath disabled: value-dependent
    ``sum(tr) == 0`` guard).
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(slow, n):
        avg_fast = np.nan
        avg_medium = np.nan
        avg_slow = np.nan
        for idx, length in ((0, fast), (1, medium), (2, slow)):
            sum_bp = 0.0
            sum_tr = 0.0
            for j in range(i - length + 1, i + 1):
                prev_close = close[j - 1]
                low_j = low[j] if low[j] < prev_close else prev_close
                bp = close[j] - low_j
                hi = high[j] if high[j] > prev_close else prev_close
                lo = low[j] if low[j] < prev_close else prev_close
                sum_bp += bp
                sum_tr += hi - lo
            avg = np.nan
            if sum_tr != 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
                avg = sum_bp / sum_tr
            if idx == 0:
                avg_fast = avg
            elif idx == 1:
                avg_medium = avg
            else:
                avg_slow = avg
        out[i] = 100.0 * (4.0 * avg_fast + 2.0 * avg_medium + avg_slow) / 7.0
    return out


def uo_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fast: int = 7,
    medium: int = 14,
    slow: int = 28,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Numpy-based Ultimate Oscillator calculation.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64), same length.
    fast, medium, slow : int
        Averaging windows (>= 1; slow gives the warm-up of ``slow``
        NaN bars).
    offset, fillna : as usual.
    use_talib : bool
        Prefer TA-Lib when available.

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    np.ndarray
        UO values in [0, 100] (NaN where a denominator is zero).

    Raises
    ------
    ValueError
        If any window < 1.

    """
    if fast < 1:
        raise ValueError("fast must be >= 1")
    if medium < 1:
        raise ValueError("medium must be >= 1")
    if slow < 1:
        raise ValueError("slow must be >= 1")
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.writeable:
        low = low.copy()
    if not close.flags.writeable:
        close = close.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if use_talib and talib_available:
        result = talib.ULTOSC(
            high,
            low,
            close,
            timeperiod1=fast,
            timeperiod2=medium,
            timeperiod3=slow,
        )
    else:
        result = _uo_numba(high, low, close, fast, medium, slow)
    return _apply_offset_fillna(result, offset, fillna)


def uo_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    fast: int = 7,
    medium: int = 14,
    slow: int = 28,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal UO (accepts numpy array or Polars Series)."""
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return uo_numpy(
        high,
        low,
        close,
        fast,
        medium,
        slow,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )


def uo_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    fast: int = 7,
    medium: int = 14,
    slow: int = 28,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add Ultimate Oscillator column to a Polars DataFrame.

    Default column name: ``UO_{fast}_{medium}_{slow}``.
    """
    high = df[high_col].cast(pl.Float64).to_numpy()
    low = df[low_col].cast(pl.Float64).to_numpy()
    close = df[close_col].cast(pl.Float64).to_numpy()
    result = uo_numpy(
        high,
        low,
        close,
        fast,
        medium,
        slow,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )
    out_name = output_col or f"UO_{fast}_{medium}_{slow}"
    return df.with_columns(pl.Series(out_name, result))
