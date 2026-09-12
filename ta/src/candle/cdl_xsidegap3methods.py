"""Upside/Downside Gap 3 Methods candlestick pattern.

This pattern identifies a bullish or bearish continuation pattern:
- Upside Gap 3 Methods (bullish): first two white candles with a gap up,
    followed by a black candle that closes into the gap.
- Downside Gap 3 Methods (bearish): first two black candles with a gap down,
    followed by a white candle that closes into the gap.

The implementation is Numba-accelerated and optionally falls back to TA-Lib
if available.

Functions:
    cdl_xsidegap3methods: Universal function (numpy or Polars Series).
    cdl_xsidegap3methods_polars: Polars DataFrame wrapper.
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
def _cdl_xsidegap3methods_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated core for Upside/Downside Gap 3 Methods.

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
        # Candle 1 (i-2)
        o1, c1 = open_[i - 2], close[i - 2]
        # Candle 2 (i-1)
        o2, c2 = open_[i - 1], close[i - 1]
        # Candle 3 (i)
        o3, c3 = open_[i], close[i]
        # Body extremes for candles 1 and 2
        b1_low = o1 if o1 < c1 else c1
        b1_high = o1 if o1 > c1 else c1
        b2_low = o2 if o2 < c2 else c2
        b2_high = o2 if o2 > c2 else c2
        # --- Upside Gap 3 Methods (bullish continuation) ---
        bull = False
        if c1 > o1 and c2 > o2 and b2_low > b1_high and c3 < o3:
            if b1_high < c3 < b2_low:
                bull = True
        # --- Downside Gap 3 Methods (bearish continuation) ---
        bear = False
        if c1 < o1 and c2 < o2 and b2_high < b1_low and c3 > o3:
            if b2_high < c3 < b1_low:
                bear = True
        if bull or bear:
            out[i] = 1
    return out


def cdl_xsidegap3methods(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Upside/Downside Gap 3 Methods pattern detection.

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
    >>> open_ = np.array([100, 115, 118], dtype=np.float64)
    >>> high = np.array([102, 120, 120], dtype=np.float64)
    >>> low = np.array([98, 114, 113], dtype=np.float64)
    >>> close = np.array([102, 118, 114], dtype=np.float64)
    >>> cdl_xsidegap3methods(open_, high, low, close)
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
        talib_out = talib.CDLXSIDEGAP3METHODS(open_, high, low, close)
        out = (talib_out != 0).astype(np.float64)
    else:
        mask = _cdl_xsidegap3methods_nb(open_, high, low, close)
        out = mask.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_xsidegap3methods_polars(
    df: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = 'CDL_XSIDEGAP3METHODS',
) -> pl.DataFrame:
    """Add Upside/Downside Gap 3 Methods pattern column to a Polars DataFrame.

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
    output_col : str, default "CDL_XSIDEGAP3METHODS"
        Name of the output column.

    Returns
    -------
    pl.DataFrame
        New DataFrame with the pattern column appended.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "open": [100, 115, 118],
    ...     "high": [102, 120, 120],
    ...     "low": [98, 114, 113],
    ...     "close": [102, 118, 114],
    ... })
    >>> cdl_xsidegap3methods_polars(df, output_col="PATTERN")
    shape: (3, 5)
    ┌──────┬──────┬──────┬───────┬─────────┐
    │ open ┆ high ┆ low  ┆ close ┆ PATTERN │
    │ ---  ┆ ---  ┆ ---  ┆ ---   ┆ ---     │
    │ f64  ┆ f64  ┆ f64  ┆ f64   ┆ f64     │
    ╞══════╪══════╪══════╪═══════╪═════════╡
    │ 100  ┆ 102  ┆ 98   ┆ 102   ┆ 0.0     │
    │ 115  ┆ 120  ┆ 114  ┆ 118   ┆ 0.0     │
    │ 118  ┆ 120  ┆ 113  ┆ 114   ┆ 1.0     │
    └──────┴──────┴──────┴───────┴─────────┘

    """
    out = cdl_xsidegap3methods(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
