"""Heikin-Ashi (HA) candles for financial time series.

Heikin-Ashi candles are a modified version of traditional candlesticks
that filter out noise and make trends easier to spot.  The calculations
use a smoothed open, high, low, and close.

This module provides Numba-accelerated computation of Heikin-Ashi values
aligned to the original bar index, with optional shifting and NaN filling.

Functions:
    ha_numpy: Calculate Heikin-Ashi from numpy arrays.
    ha: Universal Heikin-Ashi wrapper (numpy or Polars Series).
    ha_polars: Add Heikin-Ashi columns to a Polars DataFrame.

The core algorithm is implemented in Numba for high performance.
"""

import numpy as np
import polars as pl

from numba import njit

from .._array_ops import _apply_offset_fillna


@njit(
    "Tuple((float64[:], float64[:], float64[:], float64[:]))"
    "(float64[:], float64[:], float64[:], float64[:])",
    cache=True,
    fastmath=False,
)
def _heikin_ashi_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-compiled core for Heikin-Ashi calculation.

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
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Heikin-Ashi open, high, low, close as float64 arrays.

    """
    ha_close = 0.25 * (open_ + high + low + close)

    ha_open = np.empty_like(ha_close)
    ha_open[0] = 0.5 * (open_[0] + close[0])

    m = close.size
    for i in range(1, m):
        ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close[i - 1])

    ha_high = np.maximum(np.maximum(ha_open, ha_close), high)
    ha_low = np.minimum(np.minimum(ha_open, ha_close), low)

    return ha_open, ha_high, ha_low, ha_close


def ha_numpy(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Heikin-Ashi candles from numpy arrays.

    Parameters
    ----------
    open_ : np.ndarray
        Open prices (1D float64).
    high : np.ndarray
        High prices (1D float64).
    low : np.ndarray
        Low prices (1D float64).
    close : np.ndarray
        Close prices (1D float64).
    offset : int, default 0
        Shift applied to the output arrays:
        - Positive : forward shift (later values move to earlier positions)
        - Negative : backward shift (earlier values move to later positions)
        The shifted positions are filled with `fillna`
        (or NaN if not provided).
    fillna : float or None, default None
        Value used to fill positions that become NaN due to the offset.
        If None, NaN is used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Heikin-Ashi open, high, low, close as float64 arrays.

    """
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
    ha_open, ha_high, ha_low, ha_close = _heikin_ashi_nb(
        open_, high, low, close
    )
    # Apply offset and fillna
    ha_open = _apply_offset_fillna(ha_open, offset, fillna)
    ha_high = _apply_offset_fillna(ha_high, offset, fillna)
    ha_low = _apply_offset_fillna(ha_low, offset, fillna)
    ha_close = _apply_offset_fillna(ha_close, offset, fillna)
    return ha_open, ha_high, ha_low, ha_close


def ha(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Universal Heikin-Ashi wrapper (accepts numpy arrays or Polars Series).

    Parameters
    ----------
    open_ : np.ndarray or pl.Series
        Open prices.
    high : np.ndarray or pl.Series
        High prices.
    low : np.ndarray or pl.Series
        Low prices.
    close : np.ndarray or pl.Series
        Close prices.
    offset : int, default 0
        Shift applied to outputs.
    fillna : float or None, default None
        Value to fill NaNs after shifting.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Heikin-Ashi open, high, low, close as numpy arrays.

    Examples
    --------
    >>> import numpy as np
    >>> open_ = np.array([100, 102, 104, 106])
    >>> high = np.array([101, 103, 105, 107])
    >>> low = np.array([99, 101, 103, 105])
    >>> close = np.array([102, 104, 106, 108])
    >>> ha_o, ha_h, ha_l, ha_c = ha(open_, high, low, close)

    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ha_numpy(open_, high, low, close, offset, fillna)


def ha_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """Add Heikin-Ashi candle columns to a Polars DataFrame.

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
    date_col : str, default "date"
        Name of the date/time column (included in the output).
    offset : int, default 0
        Shift applied to HA values.
    fillna : float or None, default None
        Value to fill NaNs after shifting.
    suffix : str, default ""
        Suffix appended to output column names.

    Returns
    -------
    pl.DataFrame
        New DataFrame with columns:
        date_col, HA_open{suffix}, HA_high{suffix}, HA_low{suffix},
        HA_close{suffix}.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "date": [1, 2, 3, 4],
    ...         "open": [100, 102, 104, 106],
    ...         "high": [101, 103, 105, 107],
    ...         "low": [99, 101, 103, 105],
    ...         "close": [102, 104, 106, 108],
    ...     }
    ... )
    >>> ha_polars(df, suffix="_HA")
    shape: (4, 5)
    +------+------------+------------+-----------+-------------+
    | date | HA_open_HA | HA_high_HA | HA_low_HA | HA_close_HA |
    | ---  | ---        | ---        | ---       | ---         |
    | i64  | f64        | f64        | f64       | f64         |
    +======+============+============+===========+=============+
    | 1    | 101.0      | 101.0      | 99.0      | 100.5       |
    | 2    | 100.75     | 103.0      | 100.75    | 102.5       |
    | 3    | 101.625    | 105.0      | 101.625   | 104.5       |
    | 4    | 103.0625   | 107.0      | 103.0625  | 106.5       |
    +------+------------+------------+-----------+-------------+

    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    ha_open, ha_high, ha_low, ha_close = ha_numpy(
        open_arr, high_arr, low_arr, close_arr, offset, fillna
    )
    suffix = suffix or ""
    return pl.DataFrame(
        {
            date_col: df[date_col],
            f"HA_open{suffix}": ha_open,
            f"HA_high{suffix}": ha_high,
            f"HA_low{suffix}": ha_low,
            f"HA_close{suffix}": ha_close,
        }
    )
