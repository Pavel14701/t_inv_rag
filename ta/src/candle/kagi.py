"""Kagi line (yin/yang) generation for financial time series.

Kagi charts filter out small price movements and focus on significant trends.
A line changes direction only when the price moves by a specified reversal
amount from the last extreme. This module provides Numba-accelerated
computation of the Kagi line state aligned to the original bar index,
with optional shifting and NaN filling.

Functions:
    kagi: Generate Kagi yin/yang stream from price array or Polars Series.
    kagi_polars: Add Kagi yin/yang stream as a column to a Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl
from numba import njit

from .._array_ops import _apply_offset_fillna


@njit(
    'int8[:](float64[:], float64)',
    cache=True,
    fastmath=False,
)  # type: ignore[call-overload]
def _kagi_nb(prices: np.ndarray, reversal: float) -> np.ndarray:  # noqa: C901
    """Numba-accelerated Kagi line (yin/yang) aligned to bars.

    The algorithm tracks the current direction (up/down) and the last extreme
    price. The line continues in the same direction as long as new extremes
    are made. A reversal occurs only when the price moves against the current
    direction by at least `reversal` points from the last extreme.

    Parameters
    ----------
    prices : np.ndarray
        1D float64 array of prices, must be contiguous and writable.
    reversal : float
        Minimum price movement required to reverse the direction. Must be > 0.

    Returns
    -------
    np.ndarray
        int8 array of same length as `prices` with values:
        - 1  : yang (up)
        - -1 : yin (down)
        - 0  : direction not established yet

    Notes
    -----
    - The array is aligned to the input bars, so each bar has exactly one
        value (0, 1, or -1).
    - The first element is always 0 because there is no prior direction.
    - The algorithm uses strict comparisons (> for new highs, < for new lows)
        to avoid unnecessary flips.

    """
    n = prices.size
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out
    p0 = prices[0]
    direction = 0  # 0 = none, 1 = up (yang), -1 = down (yin)
    last_extreme = p0
    # Find the first direction
    for i in range(1, n):
        p = prices[i]
        if direction == 0:
            # First movement determines initial direction
            if p >= p0 + reversal:
                direction = 1
                last_extreme = p
            elif p <= p0 - reversal:
                direction = -1
                last_extreme = p
            out[i] = direction
            continue
        if direction == 1:
            # Continue yang (up) as long as we make new highs
            if p > last_extreme:
                last_extreme = p
                out[i] = 1
                continue
            # Check for reversal to yin (down)
            if p <= last_extreme - reversal:
                direction = -1
                last_extreme = p
                out[i] = -1
                continue
            # Otherwise, stay in yang
            out[i] = 1
        else:  # direction == -1
            # Continue yin (down) as long as we make new lows
            if p < last_extreme:
                last_extreme = p
                out[i] = -1
                continue
            # Check for reversal to yang (up)
            if p >= last_extreme + reversal:
                direction = 1
                last_extreme = p
                out[i] = 1
                continue
            # Otherwise, stay in yin
            out[i] = -1
    return out


def kagi(
    prices: np.ndarray | pl.Series,
    reversal: float,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Generate a Kagi yin/yang stream from price data.

    The function accepts either a NumPy array or a Polars Series,
    converts it to a contiguous float64 array, applies the Kagi algorithm,
    and optionally shifts the result and fills NaN values.

    Parameters
    ----------
    prices : np.ndarray or pl.Series
        Price sequence (e.g., close prices). If a Polars Series is given,
        it is converted to a NumPy array internally.
    reversal : float
        Minimum price movement to reverse direction. Must be > 0.
    offset : int, default 0
        Shift applied to the output array:
        - Positive : forward shift (later values move to earlier positions)
        - Negative : backward shift (earlier values move to later positions)
        The shifted positions are filled with `fillna`
        (or NaN if not provided).
    fillna : float or None, default None
        Value used to fill positions that become NaN due to the offset.
        If None, NaN is used.

    Returns
    -------
    np.ndarray
        Float64 array of the same length as the input, containing:
        - 1.0  : yang (up)
        - -1.0 : yin (down)
        - 0.0  : direction not yet established
        Values are shifted and NaN-filled according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100.0, 102.0, 104.0, 106.0])
    >>> kagi(prices, reversal=2.0)
    array([0., 1., 1., 1.])

    >>> kagi(prices, reversal=2.0, offset=1, fillna=0.0)
    array([0., 0., 1., 1.])

    """
    if isinstance(prices, pl.Series):
        prices = prices.to_numpy()
    prices = np.asarray(prices, dtype=np.float64)
    if not prices.flags.c_contiguous:
        prices = np.ascontiguousarray(prices)
    # Numba sometimes requires writable arrays
    if not prices.flags.writeable:
        prices = prices.copy()
    kagi_int = _kagi_nb(prices, float(reversal))
    out = kagi_int.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def kagi_polars(
    df: pl.DataFrame,
    price_col: str = 'close',
    reversal: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = 'KAGI',
) -> pl.DataFrame:
    """Add a Kagi yin/yang column to a Polars DataFrame.

    This is a convenience wrapper around :func:`kagi` that directly
    operates on a Polars DataFrame and returns the DataFrame with the
    new column appended.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the price column.
    price_col : str, default "close"
        Name of the column that holds the price data.
    reversal : float, default 1.0
        Minimum price movement to reverse direction.
    offset : int, default 0
        Shift applied to the output (see :func:`kagi`).
    fillna : float or None, default None
        Value to fill shifted positions (see :func:`kagi`).
    output_col : str, default "KAGI"
        Name of the column to add to the DataFrame. If a column with this
        name already exists, it will be overwritten.

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the additional Kagi column.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [100.0, 102.0, 104.0, 106.0]})
    >>> kagi_polars(df, reversal=2.0, output_col="KAGI")
    shape: (4, 2)
    ┌───────┬───────┐
    │ close ┆ KAGI  │
    │ ---   ┆ ---   │
    │ f64   ┆ f64   │
    ╞═══════╪═══════╡
    │ 100.0 ┆ 0.0   │
    │ 102.0 ┆ 1.0   │
    │ 104.0 ┆ 1.0   │
    │ 106.0 ┆ 1.0   │
    └───────┴───────┘

    """
    out = kagi(
        df[price_col].to_numpy(),
        reversal=reversal,
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
