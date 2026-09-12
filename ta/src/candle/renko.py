"""Renko brick (Renko chart) generation for financial time series.

Renko charts abstract away time and focus solely on price movement.
A new brick is drawn when the price moves by a fixed amount (box_size)
from the previous brick's closing level. This module provides Numba-
accelerated computation of Renko brick streams aligned to the original
bar index, with optional shifting and NaN filling.

Functions:
    renko: Generate Renko brick stream from price array or Polars Series.
    renko_polars: Add Renko brick stream as a column to a Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl

from numba import njit

from .._array_ops import _apply_offset_fillna


@njit(
    "int8[:](float64[:], float64)",
    cache=True,
    fastmath=False,
)  # type: ignore[call-overload]
def _renko_nb(prices: np.ndarray, box_size: float) -> np.ndarray:
    """Numba-accelerated core Renko brick generator.

    The algorithm iterates over the price array, maintaining an "anchor"
    price level that moves in whole multiples of `box_size` as the price
    moves.  Each time the price crosses a new multiple, a brick is recorded
    with the corresponding direction.

    Parameters
    ----------
    prices : np.ndarray
        1D float64 array of prices, must be contiguous and writable.
    box_size : float
        Minimum price movement required to draw a brick. Must be positive.

    Returns
    -------
    np.ndarray
        int8 array of same length as `prices` with values:
        - 1  : up brick (price crossed a new upper level)
        - -1 : down brick (price crossed a new lower level)
        - 0  : no brick (price stayed within current brick)

    Notes
    -----
    - The array is aligned to the input bars, so each bar has exactly one
        value (0, 1, or -1). This is sometimes called "Renko stream".
    - The algorithm uses a while loop for each bar to handle cases where
        the price moves by more than one box_size per bar.
    - The first element is always 0 because there is no prior anchor.

    """
    n = prices.size
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    anchor = prices[0]
    for i in range(1, n):
        p = prices[i]
        while p >= anchor + box_size:
            anchor += box_size
            out[i] = 1
        while p <= anchor - box_size:
            anchor -= box_size
            out[i] = -1
    return out


def renko(
    prices: np.ndarray | pl.Series,
    box_size: float,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Generate a Renko brick stream from price data.

    The function accepts either a NumPy array or a Polars Series,
    converts it to a contiguous float64 array, applies the Renko
    algorithm, and optionally shifts the result and fills NaN values.

    Parameters
    ----------
    prices : np.ndarray or pl.Series
        Price sequence (e.g., close prices). If a Polars Series is given,
        it is converted to a NumPy array internally.
    box_size : float
        The minimum price change to create a brick. Must be > 0.
    offset : int, default 0
        Shift applied to the output array:
        - Positive : forward shift (later values move to earlier positions)
        - Negative : backward shift (earlier values move to later positions)
        The shifted positions are filled with
        `fillna` (or NaN if not provided).
    fillna : float or None, default None
        Value used to fill positions that become NaN due to the offset.
        If None, NaN is used.

    Returns
    -------
    np.ndarray
        Float64 array of the same length as the input, containing:
        - 1.0  : up brick
        - -1.0 : down brick
        - 0.0  : no brick
        Values are shifted and NaN-filled according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100.0, 102.0, 104.0, 103.0])
    >>> renko(prices, box_size=2.0)
    array([0., 1., 1., 0.])

    >>> renko(prices, box_size=2.0, offset=1, fillna=0.0)
    array([0., 0., 1., 1.])

    """
    if isinstance(prices, pl.Series):
        prices = prices.to_numpy()
    prices = np.asarray(prices, dtype=np.float64)
    if not prices.flags.c_contiguous:
        prices = np.ascontiguousarray(prices)
    # Numba sometimes requires writable arrays
    # (e.g., for `np.ascontiguousarray`)
    if not prices.flags.writeable:
        prices = prices.copy()
    bricks = _renko_nb(prices, box_size)
    out = bricks.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def renko_polars(
    df: pl.DataFrame,
    price_col: str = "close",
    box_size: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "RENKO",
) -> pl.DataFrame:
    """Add a Renko brick stream column to a Polars DataFrame.

    This is a convenience wrapper around :func:`renko` that directly
    operates on a Polars DataFrame and returns the DataFrame with the
    new column appended.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the price column.
    price_col : str, default "close"
        Name of the column that holds the price data.
    box_size : float, default 1.0
        Minimum price movement to draw a brick.
    offset : int, default 0
        Shift applied to the output (see :func:`renko`).
    fillna : float or None, default None
        Value to fill shifted positions (see :func:`renko`).
    output_col : str, default "RENKO"
        Name of the column to add to the DataFrame. If a column with this
        name already exists, it will be overwritten.

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the additional Renko column.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [100.0, 102.0, 104.0, 103.0]})
    >>> renko_polars(df, box_size=2.0, output_col="RENKO")
    shape: (4, 2)
    +-------+-------+
    | close | RENKO |
    | ---   | ---   |
    | f64   | f64   |
    +=======+=======+
    | 100.0 | 0.0   |
    | 102.0 | 1.0   |
    | 104.0 | 1.0   |
    | 103.0 | 0.0   |
    +-------+-------+

    """
    out = renko(
        df[price_col].to_numpy(),
        box_size=box_size,
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
