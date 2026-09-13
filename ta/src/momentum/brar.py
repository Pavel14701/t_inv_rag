# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from numba import jit

from .._array_ops import _apply_offset_fillna


@jit(nopython=True, fastmath=False, cache=True)
def _window_sums(x: np.ndarray, length: int) -> np.ndarray:
    """Rolling window sums with pandas-like NaN semantics.

    A window containing any NaN yields NaN (no value emitted). When the
    last NaN leaves the window, the sum is recomputed from scratch so
    later windows recover (plain sliding sums would stay NaN forever --
    the original bug that zeroed out the whole BR line).
    """
    n = x.shape[0]
    sums = np.full(n, np.nan)
    s = 0.0
    nan_count = 0
    for i in range(n):
        v = x[i]
        if np.isnan(v):
            nan_count += 1
        if i >= length:
            u = x[i - length]
            if np.isnan(u):
                nan_count -= 1
        if nan_count > 0:
            s = np.nan  # recompute once every NaN has left the window
        else:
            if np.isnan(s):
                s = 0.0
                for k in range(i - length + 1, i + 1):
                    s += x[k]
            else:
                s += v
                if i >= length:
                    s -= x[i - length]
        if i >= length - 1 and nan_count == 0:
            sums[i] = s
    return sums


@jit(nopython=True, fastmath=False, cache=True)
def _brar_numba_core(
    high_open_range: np.ndarray,
    open_low_range: np.ndarray,
    hcy: np.ndarray,
    cyl: np.ndarray,
    length: int,
    scalar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AR and BR using sliding window sums (Numba).

    fastmath is disabled: the ``sum != 0.0`` guards are value-dependent
    IEEE-754 comparisons and sliding sums must not be re-associated.
    Zero denominators leave NaN (never inf) by design.
    AR is valid from index ``length - 1``; BR additionally waits out the
    ``drift`` NaN prefix of the shifted close (first value at
    ``length + drift - 1``).
    """
    n = len(high_open_range)
    ar = np.full(n, np.nan, dtype=np.float64)
    br = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return ar, br
    sum_high_open = _window_sums(high_open_range, length)
    sum_open_low = _window_sums(open_low_range, length)
    sum_hcy = _window_sums(hcy, length)
    sum_cyl = _window_sums(cyl, length)
    for i in range(n):
        if not np.isnan(sum_open_low[i]) and sum_open_low[i] != 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
            ar[i] = scalar * sum_high_open[i] / sum_open_low[i]
        if not np.isnan(sum_cyl[i]) and sum_cyl[i] != 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
            br[i] = scalar * sum_hcy[i] / sum_cyl[i]
    return ar, br


def brar_ind(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 26,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal BRAR indicator (always uses Numba).

    Parameters
    ----------
    open_, high, low, close : np.ndarray or pl.Series
        OHLC price series.
    length : int
        Window length.
    scalar : float
        Multiplier.
    drift : int
        Shift for close.
    offset, fillna : as usual.

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    ar, br : tuple of np.ndarray

    Raises
    ------
    ValueError
        If `length` < 1 or `drift` < 1.

    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if drift < 1:
        raise ValueError("drift must be >= 1")
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    # Ensure float64 contiguous
    open_ = np.asarray(open_, dtype=np.float64, copy=False)
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Rebind the outer names: assigning to the loop variable is a no-op
    # and left the arrays non-contiguous for the numba backend.
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Compute ranges
    high_open_range = high - open_
    open_low_range = open_ - low
    # Shifted close
    close_shifted = np.roll(close, drift)
    close_shifted[:drift] = np.nan
    hcy = np.maximum(high - close_shifted, 0.0)
    cyl = np.maximum(close_shifted - low, 0.0)
    ar, br = _brar_numba_core(
        high_open_range, open_low_range, hcy, cyl, length, scalar
    )
    ar = _apply_offset_fillna(ar, offset, fillna)
    br = _apply_offset_fillna(br, offset, fillna)
    return ar, br


def brar_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    length: int = 26,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """Returns DataFrame with date  AR, BR columns."""
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    ar, br = brar_ind(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        length=length,
        scalar=scalar,
        drift=drift,
        offset=offset,
        fillna=fillna,
    )
    suffix = suffix or f"_{length}"
    return pl.DataFrame(
        {
            date_col: df[date_col],
            f"AR{suffix}": ar,
            f"BR{suffix}": br,
        }
    )
