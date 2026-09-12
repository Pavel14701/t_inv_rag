"""Tristar candlestick pattern.

This pattern consists of three consecutive doji candles with gaps between them.
It can be a bullish or bearish reversal signal depending on the direction of
the gaps, but this implementation detects the pattern regardless of direction.

The implementation is Numba-accelerated and optionally falls back to TA-Lib
if available.

Functions:
    cdl_tristar: Universal function (numpy or Polars Series).
    cdl_tristar_polars: Polars DataFrame wrapper.
"""

import numpy as np
import polars as pl

from numba import njit

from .._array_ops import _apply_offset_fillna
from ..external import talib, talib_available


@njit(
    'int8[:](float64[:], float64[:], float64[:], float64[:])',
    cache=True,
    fastmath=False,
)
def _cdl_tristar_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated core for Tristar pattern.

    Detects the pattern at the third doji (index i). Returns an int8 array
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
        # Candle 1 (i-2)
        o1 = open_[i - 2]
        c1 = close[i - 2]
        h1 = high[i - 2]
        l1 = low[i - 2]
        # Candle 2 (i-1)
        o2 = open_[i - 1]
        c2 = close[i - 1]
        h2 = high[i - 1]
        l2 = low[i - 1]
        # Candle 3 (i)
        o3 = open_[i]
        c3 = close[i]
        h3 = high[i]
        l3 = low[i]
        # Must be doji: body extremely small (<= 10% of range)
        rng1 = h1 - l1
        rng2 = h2 - l2
        rng3 = h3 - l3
        if rng1 <= 0 or rng2 <= 0 or rng3 <= 0:
            continue
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        if body1 > 0.1 * rng1:
            continue
        if body2 > 0.1 * rng2:
            continue
        if body3 > 0.1 * rng3:
            continue
        # There must be gaps between doji
        # Gap between candle1 and candle2
        gap12 = (l2 > h1) or (h2 < l1)
        if not gap12:
            continue
        # Gap between candle2 and candle3
        gap23 = (l3 > h2) or (h3 < l2)
        if not gap23:
            continue
        # Direction (bullish/bearish) is ignored for binary output
        out[i] = 1
    return out


def cdl_tristar(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Tristar pattern detection.

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
    >>> open_ = np.array([100, 105, 110], dtype=np.float64)
    >>> high = np.array([101, 106, 111], dtype=np.float64)
    >>> low = np.array([99, 104, 109], dtype=np.float64)
    >>> close = np.array([100, 105, 110], dtype=np.float64)
    >>> cdl_tristar(open_, high, low, close)
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
        talib_out = talib.CDLTRISTAR(open_, high, low, close)
        out = (talib_out != 0).astype(np.float64)  # TA-Lib returns ±100
    else:
        mask = _cdl_tristar_nb(open_, high, low, close)
        out = mask.astype(np.float64)

    return _apply_offset_fillna(out, offset, fillna)


def cdl_tristar_polars(
    df: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = 'CDL_TRISTAR',
) -> pl.DataFrame:
    """Add Tristar pattern column to a Polars DataFrame.

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
    output_col : str, default "CDL_TRISTAR"
        Name of the output column.

    Returns
    -------
    pl.DataFrame
        New DataFrame with the pattern column appended.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "open": [100, 105, 110],
    ...     "high": [101, 106, 111],
    ...     "low": [99, 104, 109],
    ...     "close": [100, 105, 110],
    ... })
    >>> cdl_tristar_polars(df, output_col="PATTERN")
    shape: (3, 5)
    ┌──────┬──────┬──────┬───────┬─────────┐
    │ open ┆ high ┆ low  ┆ close ┆ PATTERN │
    │ ---  ┆ ---  ┆ ---  ┆ ---   ┆ ---     │
    │ f64  ┆ f64  ┆ f64  ┆ f64   ┆ f64     │
    ╞══════╪══════╪══════╪═══════╪═════════╡
    │ 100  ┆ 101  ┆ 99   ┆ 100   ┆ 0.0     │
    │ 105  ┆ 106  ┆ 104  ┆ 105   ┆ 0.0     │
    │ 110  ┆ 111  ┆ 109  ┆ 110   ┆ 1.0     │
    └──────┴──────┴──────┴───────┴─────────┘

    """
    out = cdl_tristar(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
