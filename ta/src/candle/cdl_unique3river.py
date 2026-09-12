"""Unique 3 River candlestick pattern.

This pattern is a bullish reversal pattern that appears in a downtrend.
It consists of:
1. A long black (bearish) candle.
2. A small black (bearish) candle that makes a new lower low.
3. A small white (bullish) candle that closes above the close of the
    second candle but below the open of the first candle (partial recovery).

The implementation is Numba-accelerated and optionally falls back to TA-Lib
if available.

Functions:
    cdl_unique3river: Universal function (numpy or Polars Series).
    cdl_unique3river_polars: Polars DataFrame wrapper.
"""

import numpy as np
import polars as pl

from numba import njit

from .._array_ops import _apply_offset_fillna
from ..external import talib, talib_available


@njit(
    "int8[:](float64[:], float64[:], float64[:], float64[:])",
    cache=True,
    fastmath=False,
)
def _cdl_unique3river_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated core for Unique 3 River pattern.

    Detects the pattern at the third candle (index i). Returns an int8 array
    with 1 at positions where the pattern completes, 0 otherwise.

    Parameters
    ----------
    open_ : np.ndarray
        1D float64 array of open prices.
    high : np.ndarray
        1D float64 array of high prices.
    low : np.ndarray
        1D float64 array of low prices.
    close : np.ndarray
        1D float64 array of close prices.

    Returns
    -------
    np.ndarray
        int8 array of same length as input, with 1 at pattern completion.

    """
    n = len(open_)
    out = np.zeros(n, dtype=np.int8)
    for i in range(2, n):
        # Candle 1 (i-2): long black
        o1 = open_[i - 2]
        c1 = close[i - 2]
        h1 = high[i - 2]
        l1 = low[i - 2]
        if not (c1 < o1):
            continue
        rng1 = h1 - l1
        body1 = o1 - c1
        if rng1 <= 0.0 or body1 < 0.6 * rng1:
            continue  # require a long bearish body
        # Candle 2 (i-1): small black with lower low
        o2 = open_[i - 1]
        c2 = close[i - 1]
        h2 = high[i - 1]
        l2 = low[i - 1]
        if not (c2 < o2):
            continue
        rng2 = h2 - l2
        if rng2 <= 0.0:
            continue
        body2 = o2 - c2
        # Body of candle 2 must be smaller than 40% of body1
        if body2 > 0.4 * body1:
            continue
        # Candle 2 must make a new lower low
        if l2 >= l1:
            continue
        # Candle 3 (i): small white recovery
        o3 = open_[i]
        c3 = close[i]
        h3 = high[i]
        l3 = low[i]
        if not (c3 > o3):
            continue
        rng3 = h3 - l3
        if rng3 <= 0.0:
            continue
        body3 = c3 - o3
        if body3 > 0.6 * rng3:
            continue  # must not be too long
        # Close above close2 but below open1 (partial recovery)
        if not (c3 > c2 and c3 < o1):
            continue
        # Open above low2 (no new low)
        if o3 <= l2:
            continue
        out[i] = 1
    return out


def cdl_unique3river(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Unique 3 River pattern detection.

    Parameters
    ----------
    open_ : np.ndarray or pl.Series
        1D float64 array or Polars Series of open prices.
    high : np.ndarray or pl.Series
        1D float64 array or Polars Series of high prices.
    low : np.ndarray or pl.Series
        1D float64 array or Polars Series of low prices.
    close : np.ndarray or pl.Series
        1D float64 array or Polars Series of close prices.
    offset : int, default 0
        Shift applied to the output array.
        Positive = forward shift, negative = backward shift.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.
    use_talib : bool, default True
        If True and TA-Lib is available, use TA-Lib's implementation.

    Returns
    -------
    np.ndarray
        Float64 array of same length as input, with 1.0 where pattern occurs,
        else 0.0. Shifted and NaN-filled according to `offset` and `fillna`.

    Examples
    --------
    >>> open_ = np.array([100, 92, 91], dtype=np.float64)
    >>> high = np.array([101, 93, 94], dtype=np.float64)
    >>> low = np.array([99, 89, 90], dtype=np.float64)
    >>> close = np.array([90, 91, 93], dtype=np.float64)
    >>> cdl_unique3river(open_, high, low, close)
    array([0., 0., 1.])

    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    # Ensure float64 and contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not open_.flags.writeable:
        open_ = open_.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not low.flags.writeable:
        low = low.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    if use_talib and talib_available:
        talib_out = talib.CDLUNIQUE3RIVER(open_, high, low, close)
        out = (talib_out != 0).astype(np.float64)
    else:
        mask = _cdl_unique3river_nb(open_, high, low, close)
        out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_unique3river_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "CDL_UNIQUE3RIVER",
) -> pl.DataFrame:
    """Add Unique 3 River pattern column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing OHLC columns.
    open_col : str, default "open"
        Name of the open column.
    high_col : str, default "high"
        Name of the high column.
    low_col : str, default "low"
        Name of the low column.
    close_col : str, default "close"
        Name of the close column.
    offset : int, default 0
        Shift applied to pattern.
    fillna : float or None, default None
        Value to fill NaN after shift.
    output_col : str, default "CDL_UNIQUE3RIVER"
        Name of the output column.

    Returns
    -------
    pl.DataFrame
        New DataFrame with the pattern column appended.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "open": [100, 92, 91],
    ...         "high": [101, 93, 94],
    ...         "low": [99, 89, 90],
    ...         "close": [90, 91, 93],
    ...     }
    ... )
    >>> cdl_unique3river_polars(df, output_col="PATTERN")
    shape: (3, 5)
    +------+------+------+-------+---------+
    | open | high | low  | close | PATTERN |
    | ---  | ---  | ---  | ---   | ---     |
    | f64  | f64  | f64  | f64   | f64     |
    +======+======+======+=======+=========+
    | 100  | 101  | 99   | 90    | 0.0     |
    | 92   | 93   | 89   | 91    | 0.0     |
    | 91   | 94   | 90   | 93    | 1.0     |
    +------+------+------+-------+---------+

    """
    out = cdl_unique3river(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
