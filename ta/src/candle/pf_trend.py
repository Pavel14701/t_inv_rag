"""Point & Figure (P&F) trend state (X/O columns) for financial time series.

P&F charts filter out small price movements and focus on significant trends.
This module provides Numba-accelerated computation of the P&F trend state
aligned to the original bar index, with optional shifting and NaN filling.

Functions:
    pf_trend: Generate P&F trend state from price array or Polars Series.
    pf_trend_polars: Add P&F trend state as a column to a Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl
from numba import njit

from .._array_ops import _apply_offset_fillna


@njit(
    'int8[:](float64[:], float64, int64)',
    cache=True,
    fastmath=False,
)  # type: ignore[call-overload]
def _pf_trend_nb(  # noqa: C901
    prices: np.ndarray,
    box_size: float,
    reversal: int
) -> np.ndarray:
    """Numba-accelerated Point & Figure trend state (X/O columns).

    The algorithm maintains a current column direction (X = up, O = down).
    A new column is started when the price moves by at least `box_size` from
    the first price.  Once a column is established, it continues until a
    reversal of `reversal * box_size` occurs.

    Parameters
    ----------
    prices : np.ndarray
        1D float64 array of prices, must be contiguous and writable.
    box_size : float
        Minimum price movement to draw a new box. Must be positive.
    reversal : int
        Number of boxes required to reverse the column direction.
        Must be >= 1 (typically 3).

    Returns
    -------
    np.ndarray
        int8 array of same length as `prices` with values:
        - 1  : X column (up)
        - -1 : O column (down)
        - 0  : no column established yet (first bar)

    Notes
    -----
    - The array is aligned to the input bars, so each bar has exactly one
        value (0, 1, or -1).
    - The first element is always 0 because there is no prior column.
    - The algorithm uses floor division to compute box levels.

    """
    n = prices.size
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out
    p0 = prices[0]
    cur_kind = 0  # 0 = none, 1 = X (up), -1 = O (down)
    col_top = p0
    col_bottom = p0
    # Find the first column
    for i in range(1, n):
        p = prices[i]
        if cur_kind == 0:
            # No column yet: check if price moves up or
            # down by at least box_size
            if p >= p0 + box_size:
                cur_kind = 1
                col_bottom = np.floor(p0 / box_size) * box_size
                col_top = np.floor(p / box_size) * box_size
            elif p <= p0 - box_size:
                cur_kind = -1
                col_top = np.floor(p0 / box_size) * box_size
                col_bottom = np.floor(p / box_size) * box_size
            out[i] = cur_kind
            continue
        if cur_kind == 1:
            # Continue X column upward
            needed_up = col_top + box_size
            if p >= needed_up:
                col_top = np.floor(p / box_size) * box_size
                out[i] = 1
                continue
            # Check for reversal to O (down)
            rev_level = col_top - box_size * reversal
            if p <= rev_level:
                cur_kind = -1
                col_bottom = np.floor(p / box_size) * box_size
                out[i] = -1
                continue
            out[i] = 1
        else:  # cur_kind == -1
            # Continue O column downward
            needed_down = col_bottom - box_size
            if p <= needed_down:
                col_bottom = np.floor(p / box_size) * box_size
                out[i] = -1
                continue
            # Check for reversal to X (up)
            rev_level = col_bottom + box_size * reversal
            if p >= rev_level:
                cur_kind = 1
                col_top = np.floor(p / box_size) * box_size
                out[i] = 1
                continue
            out[i] = -1
    return out


def pf_trend(
    prices: np.ndarray | pl.Series,
    box_size: float,
    reversal: int = 3,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """Generate Point & Figure trend state from price data.

    The function accepts either a NumPy array or a Polars Series,
    converts it to a contiguous float64 array, applies the P&F algorithm,
    and optionally shifts the result and fills NaN values.

    Parameters
    ----------
    prices : np.ndarray or pl.Series
        Price sequence (e.g., close prices). If a Polars Series is given,
        it is converted to a NumPy array internally.
    box_size : float
        Minimum price movement to draw a box. Must be > 0.
    reversal : int, default 3
        Number of boxes required to reverse the column direction.
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
        - 1.0  : X column (up)
        - -1.0 : O column (down)
        - 0.0  : no column yet
        Values are shifted and NaN-filled according to `offset` and `fillna`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100.0, 102.0, 104.0, 106.0])
    >>> pf_trend(prices, box_size=2.0, reversal=3)
    array([0., 1., 1., 1.])

    >>> pf_trend(prices, box_size=2.0, reversal=3, offset=1, fillna=0.0)
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

    trend_int = _pf_trend_nb(prices, float(box_size), int(reversal))
    out = trend_int.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def pf_trend_polars(
    df: pl.DataFrame,
    price_col: str = 'close',
    box_size: float = 1.0,
    reversal: int = 3,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = 'PF_TREND',
) -> pl.DataFrame:
    """Add a Point & Figure trend column to a Polars DataFrame.

    This is a convenience wrapper around :func:`pf_trend` that directly
    operates on a Polars DataFrame and returns the DataFrame with the
    new column appended.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the price column.
    price_col : str, default "close"
        Name of the column that holds the price data.
    box_size : float, default 1.0
        Minimum price movement to draw a box.
    reversal : int, default 3
        Number of boxes required to reverse the column direction.
    offset : int, default 0
        Shift applied to the output (see :func:`pf_trend`).
    fillna : float or None, default None
        Value to fill shifted positions (see :func:`pf_trend`).
    output_col : str, default "PF_TREND"
        Name of the column to add to the DataFrame. If a column with this
        name already exists, it will be overwritten.

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the additional P&F trend column.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"close": [100.0, 102.0, 104.0, 106.0]})
    >>> pf_trend_polars(df, box_size=2.0, reversal=3, output_col="PF_TREND")
    shape: (4, 2)
    ┌───────┬───────────┐
    │ close ┆ PF_TREND  │
    │ ---   ┆ ---       │
    │ f64   ┆ f64       │
    ╞═══════╪═══════════╡
    │ 100.0 ┆ 0.0       │
    │ 102.0 ┆ 1.0       │
    │ 104.0 ┆ 1.0       │
    │ 106.0 ┆ 1.0       │
    └───────┴───────────┘

    """
    out = pf_trend(
        df[price_col].to_numpy(),
        box_size=box_size,
        reversal=reversal,
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
