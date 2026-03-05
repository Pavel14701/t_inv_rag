# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# True Range – Numba core
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _true_range_numba_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int,
    prenan: bool
) -> np.ndarray:
    """
    Compute True Range (TR) using Numba.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64).
    drift : int
        Shift for previous close (usually 1).
    prenan : bool
        If True, set first `drift` values to NaN; otherwise keep computed
        values (which may be based on incomplete data).

    Returns
    -------
    np.ndarray
        TR array.
    """
    n = len(high)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n <= drift:
        return tr
    for i in range(drift, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - drift])
        lc = abs(low[i] - close[i - drift])
        tr[i] = max(hl, hc, lc)
    if not prenan:
        # For indices < drift, compute using available data? In original
        # pandas_ta, when prenan=False, values are computed for all indices,
        # but for i < drift, the formula uses close[i-drift] which doesn't
        # exist, resulting in potentially bogus values. However, to match
        # the original behaviour, we keep them as computed (they will be
        # calculated in the loop above only for i>=drift, so earlier indices
        # remain NaN). Actually, the loop only fills from drift onward.
        # So if prenan=False, we still have NaN for i<drift. That's consistent
        # with pandas_ta? In pandas_ta, when prenan=False, the values are
        # computed for all indices (using shift which returns NaN for missing),
        # so the first drift values become NaN anyway. So both settings yield
        # NaN for first drift. The difference is that with prenan=True, they
        # are explicitly set to NaN after calculation, but they are already NaN.
        # The original code: after computing via concat/max, they do
        # if prenan: true_range.iloc[:drift] = nan. So it's redundant if they
        # were already NaN. But for safety, we follow the exact logic:
        # if prenan is False, we leave as is (already NaN for first drift).
        # So nothing extra needed.
        pass
    return tr


def true_range_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int = 1,
    prenan: bool = False,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    True Range using Numba.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays.
    drift : int
        Shift for previous close.
    prenan : bool
        If True, explicitly set first `drift` values to NaN (already NaN by default).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        TR values.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Ensure contiguous
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    tr = _true_range_numba_core(high, low, close, drift, prenan)
    return _apply_offset_fillna(tr, offset, fillna)


# ----------------------------------------------------------------------
# True Range – TA-Lib wrapper
# ----------------------------------------------------------------------
def true_range_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int = 1,
    prenan: bool = False,   # ignored, TA-Lib always returns NaN for first period
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    True Range using TA-Lib (C implementation).

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays.
    drift : int
        Ignored (TA-Lib uses fixed drift=1).
    prenan : bool
        Ignored (TA-Lib always returns NaN for first period).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        TR values.
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
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
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal True Range with automatic backend selection.

    Parameters
    ----------
    high, low, close : np.ndarray or pl.Series
        Price series.
    drift : int
        Shift for previous close.
    prenan : bool
        If True, explicitly set first `drift` values to NaN (Numba only).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        If True and TA-Lib is available, use it; else use Numba.

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
    if use_talib and talib_available:
        return true_range_talib(high, low, close, drift, prenan, offset, fillna)
    else:
        return true_range_numba(high, low, close, drift, prenan, offset, fillna)


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
    output_col: str | None = None
) -> pl.DataFrame:
    """
    True Range for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col, low_col, close_col : str
        Column names for prices.
    drift : int
        Shift for previous close.
    prenan : bool
        If True, explicitly set first `drift` values to NaN.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA-Lib if available.
    output_col : str, optional
        Output column name (default f"TRUERANGE_{drift}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.    
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = true_range_ind(
        high, low, close,
        drift=drift,
        prenan=prenan,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib
    )
    out_name = output_col or f"TRUERANGE_{drift}"
    return df.with_columns([pl.Series(out_name, result)])